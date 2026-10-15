#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,1024 128,1024 64,16384 64,32768 128,8192 128,16384}"
BACKEND="${BACKEND:-cuda}"

make -C futhark compile BACKEND="$BACKEND"
mkdir -p data

for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"
  futhark dataset -b -s 0 --f32-bounds=-1:1 \
    -g "[$((n / d))][$d][$d]f32" -g "[$n][$d]f32" -g "[$n][$d]f32" > "$f.in"
  "futhark/build/custom-alg1-opt-$BACKEND" --entry-point="main$d" -b < "$f.in" > "$f.out"
  echo "wrote $f.in $f.out"
done
