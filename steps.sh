#!/usr/bin/env bash
# Benchmark every step of the optimization history of attention-cuda/.
#
# Each step's kernel is rebuilt straight from its git commit (no checkout
# needed), run on the random datasets in data/, checked against the Futhark
# reference outputs with `futhark datacmp`, and timed. The CUDA baseline in
# original-cuda/ is measured in the same run for reference.
#
# Every version is built and run exactly like compare.sh does it: the same nvcc
# flags, the baseline built with `make -C original-cuda compile`, the same data
# and the same correctness check.
#
# Usage: ./steps.sh [d,N ...]       default: the four benchmark sizes
# Env:   RUNS=10 (runs per measurement), ROUNDS=3, DEVICE=0
# Needs: git with the commits below (git fetch), nvcc, futhark (module load).
# Output: per-size tables and a per-section summary on stdout;
#         results/steps.tsv (best median per version), results/steps-raw.tsv (every round).
#
# Measurements can be disturbed (the GPUs run without persistence mode, and a whole
# measurement occasionally runs ~1.5-1.8x slower). To be robust against this, the
# script keeps the driver loaded with a background nvidia-smi, warms up before every
# measurement, measures all versions ROUNDS times in turn, and reports for every
# version the lowest of its round medians, together with the spread between rounds.
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,16384 64,32768 128,8192 128,16384}"
RUNS="${RUNS:-10}"
ROUNDS="${ROUNDS:-3}"
GPU="${DEVICE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID       # same GPU numbering as nvidia-smi
export CUDA_VISIBLE_DEVICES="$GPU"

NVCCFLAGS=(-O3 -diag-suppress=1650)      # same as compare.sh

# Report sections, in order.
SECTION_TITLES=(
  "Baseline"
  "Naive implementation"
  "Switching the loop order"
  "Avoiding shared-memory bank conflicts"
  "Register tiling"
  "Online softmax in registers"
  "Reducing register and shared-memory usage"
  "Unrolling and occupancy"
  "Swizzled shared-memory layouts"
  "Overlapping loads with computation"
)

# section | commit of attention-cuda/main.cu | step
STEPS=(
  "1|0e114a0|Naive implementation"
  "2|06a1b5c|Switch loop order"
  "3|bafd852|Pad K tile"
  "4|7774702|Register tiling of Q K^T"
  "4|39b8993|Register tiling of P V, O in registers"
  "5|86356b1|Online softmax in registers"
  "6|cb90413|Reduce register pressure in O update"
  "6|993c7d7|Alias K and V in shared memory"
  "7|5eda946|Unroll fixed-trip-count loops"
  "7|1dbb008|Cap thread blocks per SM"
  "7|978f0ce|(timing only: CUDA events, same kernel)"
  "7|9d9285e|Keep k-loops rolled"
  "8|595a3f9|Swizzled shared-memory layouts"
  "9|6fe99d7|Prefetch V with cp.async"
)

BUILD=build/steps
TSV=results/steps.tsv
RAW=results/steps-raw.tsv
mkdir -p "$BUILD" results data

for step in "${STEPS[@]}"; do
  IFS='|' read -r _ commit _ <<< "$step"
  if ! git cat-file -e "$commit^{commit}" 2>/dev/null; then
    echo "Commit $commit not found; run 'git fetch' first." >&2
    exit 1
  fi
done

# ---- Build -------------------------------------------------------------------
# Baseline exactly as compare.sh builds it.
make -C original-cuda compile
ORIG=original-cuda/flash_attention_gpu

echo "Building ${#STEPS[@]} versions (nvcc ${NVCCFLAGS[*]})..." >&2
pids=()
for step in "${STEPS[@]}"; do
  IFS='|' read -r _ commit _ <<< "$step"
  dir="$BUILD/$commit"
  (
    rm -rf "$dir" && mkdir -p "$dir"
    git archive "$commit" attention-cuda utils | tar -x -C "$dir"
    nvcc "${NVCCFLAGS[@]}" -o "$dir/attention" "$dir/attention-cuda/main.cu" > "$dir/build.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true      # a failed build shows up as NOBUILD in the tables
done

# ---- Run ---------------------------------------------------------------------
# Keep the driver loaded between the programs (no persistence mode on these GPUs).
nvidia-smi -i "$GPU" -l 1 > /dev/null 2>&1 &
KEEPALIVE=$!
trap 'kill "$KEEPALIVE" 2>/dev/null || true' EXIT

# Median of a file with one runtime in seconds per line, printed in ms.
median_ms() {
  sort -g "$1" | awk '{ a[NR] = $1 }
    END { if (NR == 0) { print "-"; exit }
          m = NR % 2 ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
          printf "%.3f", m * 1000 }'
}

# run <exe> <d> <N> <dataset prefix> <tag> <check: 1|0>  ->  "<median ms> <OK|FAIL|-|CRASH|NOBUILD>"
run() {
  local exe="$1" d="$2" n="$3" f="$4" out="$BUILD/$5-d$2-N$3" check="$6"
  if [ ! -x "$exe" ]; then echo "- NOBUILD"; return; fi
  "$exe" "$d" "$n" 3 "$f.in" "$out.res" > /dev/null 2>&1 || true      # warm-up
  if ! "$exe" "$d" "$n" "$RUNS" "$f.in" "$out.res" > "$out.times" 2> "$out.log"; then
    echo "- CRASH"; return
  fi
  local ok=-
  if [ "$check" = 1 ]; then
    ok=OK
    futhark datacmp "$out.res" "$f.out" > "$out.cmp" 2>&1 || ok=FAIL
  fi
  rm -f "$out.res"
  echo "$(median_ms "$out.times") $ok"
}

printf 'd\tN\tsection\tsection_title\tcommit\tstep\tround\tms\tcheck\n' > "$RAW"
for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"
  [ -f "$f.in" ] && [ -f "$f.out" ] || ./gen-datasets.sh "$size"

  for round in $(seq "$ROUNDS"); do
    check=$([ "$round" = 1 ] && echo 1 || echo 0)    # correctness is checked in the first round
    echo "d=$d N=$n, round $round/$ROUNDS" >&2

    read -r ms ok < <(run "$ORIG" "$d" "$n" "$f" original "$check")
    printf '%s\t%s\t0\t%s\tbaseline\tCUDA baseline\t%s\t%s\t%s\n' \
      "$d" "$n" "${SECTION_TITLES[0]}" "$round" "$ms" "$ok" >> "$RAW"

    for step in "${STEPS[@]}"; do
      IFS='|' read -r section commit label <<< "$step"
      echo "  $commit $label" >&2
      read -r ms ok < <(run "$BUILD/$commit/attention" "$d" "$n" "$f" "$commit" "$check")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$d" "$n" "$section" "${SECTION_TITLES[$section]}" "$commit" "$label" "$round" "$ms" "$ok" >> "$RAW"
    done
  done
done

# Per version: lowest round median, check from the first round, spread = max / min - 1.
awk -F'\t' -v OFS='\t' '
  NR == 1 { print "d", "N", "section", "section_title", "commit", "step", "ms", "check", "spread"; next }
  {
    key = $1 OFS $2 OFS $3 OFS $4 OFS $5 OFS $6
    if (!(key in seen)) { seen[key] = 1; order[++n] = key }
    if ($9 != "-") check[key] = $9
    if ($8 != "-") {
      if (!(key in lo) || $8 + 0 < lo[key]) lo[key] = $8 + 0
      if (!(key in hi) || $8 + 0 > hi[key]) hi[key] = $8 + 0
    }
  }
  END {
    for (i = 1; i <= n; i++) {
      k = order[i]
      if (k in lo) print k, sprintf("%.3f", lo[k]), check[k], sprintf("%.0f%%", 100 * (hi[k] / lo[k] - 1))
      else         print k, "-", (k in check ? check[k] : "CRASH"), "-"
    }
  }
' "$RAW" > "$TSV"

# ---- Report ------------------------------------------------------------------
awk -F'\t' '
  function x(a, b) { return (a == "-" || b == "-" || b == 0) ? "-" : sprintf("%.2fx", a / b) }
  NR == 1 { next }
  {
    key = $1 "," $2
    if (!(key in nrows)) order[++nkeys] = key
    i = ++nrows[key]
    sec[key, i] = $3; com[key, i] = $5; lab[key, i] = $6; ms[key, i] = $7; chk[key, i] = $8; spr[key, i] = $9
    if ($3 > 0) {
      title[$3] = $4; if ($3 > maxsec) maxsec = $3
      if ($7 != "-") last[key, $3] = $7
    }
  }
  END {
    for (k = 1; k <= nkeys; k++) {
      key = order[k]; split(key, dn, ",")
      printf "\n### d = %s, N = %s\n\n", dn[1], dn[2]
      print "| Sec | Step | Commit | ms | Spread | Check | vs previous | vs naive | vs baseline |"
      print "| --: | :-- | :-- | --: | --: | :-: | --: | --: | --: |"
      base = naive = prev = "-"
      for (i = 1; i <= nrows[key]; i++) {
        m = ms[key, i]
        if (sec[key, i] == 0) {
          base = m
          printf "| - | %s | - | %s | %s | %s | - | - | 1.00x |\n", lab[key, i], m, spr[key, i], chk[key, i]
          continue
        }
        if (sec[key, i] == 1) naive = m
        printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", sec[key, i], lab[key, i], com[key, i], \
               m, spr[key, i], chk[key, i], x(prev, m), x(naive, m), x(base, m)
        if (m != "-") prev = m
      }
    }

    printf "\n### Per section: median ms at the end of the section (speedup over the previous section)\n\n"
    hdr = "| Sec | Section |"; sep = "| --: | :-- |"
    for (k = 1; k <= nkeys; k++) {
      split(order[k], dn, ","); hdr = hdr " d=" dn[1] " N=" dn[2] " |"; sep = sep " --: |"
    }
    print hdr; print sep
    for (s = 1; s <= maxsec; s++) {
      row = "| " s " | " title[s] " |"
      for (k = 1; k <= nkeys; k++) {
        key = order[k]
        cur = ((key, s) in last) ? last[key, s] : "-"
        before = ((key, s - 1) in last) ? last[key, s - 1] : "-"
        row = row " " cur (s > 1 ? " (" x(before, cur) ")" : "") " |"
      }
      print row
    }
    print "\nms: lowest of the per-round medians. Spread: how much slower the slowest round was (large = disturbed)."
    print "Checks: futhark datacmp against data/*.out, first round."
    print "Steps before 978f0ce are timed with gettimeofday + cudaDeviceSynchronize, later ones with CUDA events;"
    print "the 978f0ce row (same kernel as 1dbb008) shows the size of that difference."
  }
' "$TSV"
