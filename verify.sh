#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,1024 128,1024 64,16384 64,32768 128,8192 128,16384}"
status=0

if [ -z "${EXE:-}" ]; then
  make -C cuda compile
  EXE=cuda/flash_attention_gpu
fi

for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"
  if "$EXE" "$d" "$n" 1 "$f.in" "$f.res" > /dev/null &&
     futhark datacmp "$f.res" "$f.out" 2>&1 | head -n 5; then
    echo "d=$d N=$n OK"
  else
    echo "d=$d N=$n FAIL"
    status=1
  fi
done

exit "$status"
