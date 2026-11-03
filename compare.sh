#!/usr/bin/env bash
# Compare correctness and performance of the three implementations on data/:
#   original  original-cuda/flash_attention_gpu
#   tensor    attention-cuda/main_tensor.cu (TF32 tensor cores, needs sm_80)
#   futhark   futhark/custom-alg1-opt.fut (also produces the reference outputs)
# Usage: ./compare.sh [d,N ...]   (env: RUNS=10, ROUNDS=3, DEVICE=0)
#
# Measured like steps.sh. In every round, each program does a short warm-up run
# and then RUNS timed runs. The table shows, per program, the lowest of the round
# medians, and the spread between the rounds. Correctness is checked in round 1.
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,1024 128,1024 64,16384 64,32768 128,8192 128,16384}"
RUNS="${RUNS:-10}"
ROUNDS="${ROUNDS:-3}"
GPU="${DEVICE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID       # same GPU numbering as nvidia-smi
export CUDA_VISIBLE_DEVICES="$GPU"

ORIG=original-cuda/flash_attention_gpu
TENSOR=attention-cuda/attention_tensor
FUT=futhark/build/custom-alg1-opt-cuda

make -C original-cuda compile
make -C futhark compile BACKEND=cuda
nvcc -O3 -diag-suppress=1650 -arch=sm_80 -o "$TENSOR" attention-cuda/main_tensor.cu

# Median of a file with one number per line, scaled by $2.
median() {
  sort -g "$1" | awk -v s="$2" '{ a[NR] = $1 }
    END { if (NR == 0) { print "-"; exit }
          m = NR % 2 ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
          printf "%.3f", m * s }'
}

# Run a CUDA executable on one dataset in one round.
# run_cuda <exe> <d> <N> <dataset prefix> <tag> <round>  ->  "<median ms> <OK|FAIL|-|CRASH>"
run_cuda() {
  local exe="$1" d="$2" n="$3" f="$4" tag="$5" round="$6"
  local times="$f.$tag.$round.times" res="$f.$tag.res"
  "$exe" "$d" "$n" 3 "$f.in" "$res" > /dev/null 2>&1 || true      # warm-up
  if ! "$exe" "$d" "$n" "$RUNS" "$f.in" "$res" > "$times" 2> "$f.$tag.log"; then
    echo "- CRASH"
    return
  fi
  local ok=-
  if [ "$round" = 1 ]; then
    ok=OK
    futhark datacmp "$res" "$f.out" > "$f.$tag.cmp" 2>&1 || ok=FAIL
  fi
  echo "$(median "$times" 1000) $ok"
}

# Run Futhark on one dataset in one round; print the median in ms.
# -t records one runtime (microseconds) per run.
run_fut() {
  local d="$1" f="$2" round="$3"
  "$FUT" --entry-point="main$d" -b -r 3 < "$f.in" > /dev/null      # warm-up
  "$FUT" --entry-point="main$d" -b -r "$RUNS" -t "$f.fut.$round.times" < "$f.in" > /dev/null
  median "$f.fut.$round.times" 0.001
}

# Lowest of the round medians, and how much slower the slowest round was.
# best <ms> ...  ->  "<ms> <spread>"
best() {
  printf '%s\n' "$@" | awk '$1 != "-" { v = $1 + 0; if (n++ == 0 || v < lo) lo = v; if (n == 1 || v > hi) hi = v }
    END { if (n == 0) print "- -"; else printf "%.3f %.0f%%\n", lo, 100 * (hi / lo - 1) }'
}

mkdir -p data
for size in $SIZES; do
  f="data/d${size%,*}-N${size#*,}"
  [ -f "$f.in" ] && [ -f "$f.out" ] || ./gen-datasets.sh "$size"
done

# The GPUs run without persistence mode, and an idle GPU needs a few seconds of
# load to reach full speed (the first runs can be up to 1.8x slower). Keep the
# driver loaded, and warm the GPU up with the baseline before measuring anything.
nvidia-smi -i "$GPU" -l 1 > /dev/null 2>&1 &
KEEPALIVE=$!
trap 'kill "$KEEPALIVE" 2>/dev/null || true' EXIT
first="${SIZES%% *}"
warm="data/d${first%,*}-N${first#*,}"
end=$((SECONDS + 5))
while [ "$SECONDS" -lt "$end" ]; do
  "$ORIG" "${first%,*}" "${first#*,}" "$RUNS" "$warm.in" "$warm.warmup.res" > /dev/null 2>&1 || true
done
rm -f "$warm.warmup.res"

rows=()
for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"

  om=() tm=() fm=()
  for round in $(seq "$ROUNDS"); do
    echo "d=$d N=$n, round $round/$ROUNDS" >&2
    read -r ms ok < <(run_cuda "$ORIG" "$d" "$n" "$f" orig "$round")
    om+=("$ms")
    if [ "$round" = 1 ]; then orig_ok="$ok"; fi
    read -r ms ok < <(run_cuda "$TENSOR" "$d" "$n" "$f" tensor "$round")
    tm+=("$ms")
    if [ "$round" = 1 ]; then tensor_ok="$ok"; fi
    fm+=("$(run_fut "$d" "$f" "$round")")
  done

  read -r orig_ms orig_spread < <(best "${om[@]}")
  read -r tensor_ms tensor_spread < <(best "${tm[@]}")
  read -r fut_ms fut_spread < <(best "${fm[@]}")
  rows+=("$d $n $orig_ms $orig_spread $orig_ok $tensor_ms $tensor_spread $tensor_ok $fut_ms $fut_spread")
done

printf '\n| %3s | %6s | %13s | %6s | %8s | %11s | %6s | %6s | %12s | %6s |\n' \
  d N "original (ms)" spread original "tensor (ms)" spread tensor "futhark (ms)" spread
printf '| --: | -----: | ------------: | -----: | :------: | ----------: | -----: | :----: | -----------: | -----: |\n'
status=0
for row in "${rows[@]}"; do
  read -r d n om os ook tm ts tok fm fs <<< "$row"
  printf '| %3s | %6s | %13s | %6s | %8s | %11s | %6s | %6s | %12s | %6s |\n' \
    "$d" "$n" "$om" "$os" "$ook" "$tm" "$ts" "$tok" "$fm" "$fs"
  [ "$ook" = OK ] && [ "$tok" = OK ] || status=1
done
echo
echo "ms: lowest of the per-round medians ($ROUNDS rounds of $RUNS runs). Spread: how much slower the slowest round was (large = disturbed)."
echo "Correctness is checked with futhark datacmp against data/*.out (Futhark reference), in round 1."
echo "Mismatch details: data/*.{orig,tensor}.cmp, logs: data/*.{orig,tensor}.log, runtimes: data/*.<round>.times"
exit "$status"
