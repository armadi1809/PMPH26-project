#!/usr/bin/env bash
# Compare correctness and performance of the three implementations on data/:
#   original  original-cuda/flash_attention_gpu
#   attention attention-cuda/main.cu
#   futhark   futhark/custom-alg1-opt.fut (also produces the reference outputs)
# Usage: ./compare.sh [d,N ...]   (env: RUNS=10, DEVICE=0)
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,1024 128,1024 64,16384 64,32768 128,8192 128,16384}"
RUNS="${RUNS:-10}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"

ORIG=original-cuda/flash_attention_gpu
ATTN=attention-cuda/attention
FUT=futhark/build/custom-alg1-opt-cuda

make -C original-cuda compile
make -C futhark compile BACKEND=cuda
nvcc -O3 -diag-suppress=1650 -o "$ATTN" attention-cuda/main.cu

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
  if ! "$exe" "$d" "$n" "$RUNS" "$f.in" "$res" > "$times" 2> "$f.$tag.log"; then
    echo "- CRASH"
    return
  fi
  local ok=OK
  futhark datacmp "$res" "$f.out" > "$f.$tag.cmp" 2>&1 || ok=FAIL
  echo "$(median "$times" 1000) $ok"
}

mkdir -p data
rows=()
for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"
  [ -f "$f.in" ] && [ -f "$f.out" ] || ./gen-datasets.sh "$size"

  echo "d=$d N=$n" >&2
  read -r orig_ms orig_ok < <(run_cuda "$ORIG" "$d" "$n" "$f" orig)
  read -r attn_ms attn_ok < <(run_cuda "$ATTN" "$d" "$n" "$f" attn)

  # Futhark -t records one runtime (microseconds) per run.
  "$FUT" --entry-point="main$d" -b -r "$RUNS" -t "$f.fut.times" < "$f.in" > /dev/null
  fut_ms="$(median "$f.fut.times" 0.001)"

  rows+=("$d $n $orig_ms $orig_ok $attn_ms $attn_ok $fut_ms")
done

printf '\n| %3s | %6s | %13s | %8s | %14s | %8s | %12s |\n' \
  d N "original (ms)" "original" "attention (ms)" "attention" "futhark (ms)"
printf '| --: | -----: | ------------: | :------: | -------------: | :------: | -----------: |\n'
status=0
for row in "${rows[@]}"; do
  read -r d n om ook am aok fm <<< "$row"
  printf '| %3s | %6s | %13s | %8s | %14s | %8s | %12s |\n' "$d" "$n" "$om" "$ook" "$am" "$aok" "$fm"
  [ "$ook" = OK ] && [ "$aok" = OK ] || status=1
done
echo
echo "Correctness is checked with futhark datacmp against data/*.out (Futhark reference)."
echo "Medians over $RUNS runs. Mismatch details: data/*.{orig,attn}.cmp, logs: data/*.{orig,attn}.log"
exit "$status"
