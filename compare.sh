#!/usr/bin/env bash
# Compare correctness and performance of the three implementations on data/:
#   original  original-cuda/flash_attention_gpu
#   tensor    attention-cuda/main_tensor.cu (TF32 tensor cores, needs sm_80)
#   futhark   futhark/custom-alg1-opt.fut (also produces the reference outputs)
# Usage: ./compare.sh [d,N ...]   (env: RUNS=10, DEVICE=0)
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,1024 128,1024 64,16384 64,32768 128,8192 128,16384}"
RUNS="${RUNS:-10}"
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

# Run a CUDA executable on one dataset; print "<median ms> <OK|FAIL>".
run_cuda() {
  local exe="$1" d="$2" n="$3" f="$4" tag="$5"
  local times="$f.$tag.times" res="$f.$tag.res"
  "$exe" "$d" "$n" 3 "$f.in" "$res" > /dev/null 2>&1 || true      # warm-up
  if ! "$exe" "$d" "$n" "$RUNS" "$f.in" "$res" > "$times" 2> "$f.$tag.log"; then
    echo "- CRASH"
    return
  fi
  local ok=OK
  futhark datacmp "$res" "$f.out" > "$f.$tag.cmp" 2>&1 || ok=FAIL
  echo "$(median "$times" 1000) $ok"
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

  echo "d=$d N=$n" >&2
  read -r orig_ms orig_ok < <(run_cuda "$ORIG" "$d" "$n" "$f" orig)
  read -r tensor_ms tensor_ok < <(run_cuda "$TENSOR" "$d" "$n" "$f" tensor)

  # Futhark -t records one runtime (microseconds) per run.
  "$FUT" --entry-point="main$d" -b -r "$RUNS" -t "$f.fut.times" < "$f.in" > /dev/null
  fut_ms="$(median "$f.fut.times" 0.001)"

  rows+=("$d $n $orig_ms $orig_ok $tensor_ms $tensor_ok $fut_ms")
done

printf '\n| %3s | %6s | %13s | %8s | %11s | %6s | %12s |\n' \
  d N "original (ms)" "original" "tensor (ms)" "tensor" "futhark (ms)"
printf '| --: | -----: | ------------: | :------: | ----------: | :----: | -----------: |\n'
status=0
for row in "${rows[@]}"; do
  read -r d n om ook tm tok fm <<< "$row"
  printf '| %3s | %6s | %13s | %8s | %11s | %6s | %12s |\n' "$d" "$n" "$om" "$ook" "$tm" "$tok" "$fm"
  [ "$ook" = OK ] && [ "$tok" = OK ] || status=1
done
echo
echo "Correctness is checked with futhark datacmp against data/*.out (Futhark reference)."
echo "Medians over $RUNS runs. Mismatch details: data/*.{orig,tensor}.cmp, logs: data/*.{orig,tensor}.log"
exit "$status"
