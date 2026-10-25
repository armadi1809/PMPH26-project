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
# Env:   RUNS=10 (runs per version), DEVICE=0
# Needs: git with the commits below (git fetch), nvcc, futhark (module load).
# Output: per-size tables and a per-section summary on stdout;
#         raw medians in results/steps.tsv.
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${*:-64,16384 64,32768 128,8192 128,16384}"
RUNS="${RUNS:-10}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"

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
# Median of a file with one runtime in seconds per line, printed in ms.
median_ms() {
  sort -g "$1" | awk '{ a[NR] = $1 }
    END { if (NR == 0) { print "-"; exit }
          m = NR % 2 ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
          printf "%.3f", m * 1000 }'
}

# run <exe> <d> <N> <dataset prefix> <tag>  ->  "<median ms> <OK|FAIL|CRASH|NOBUILD>"
run() {
  local exe="$1" d="$2" n="$3" f="$4" out="$BUILD/$5-d$2-N$3"
  if [ ! -x "$exe" ]; then echo "- NOBUILD"; return; fi
  if ! "$exe" "$d" "$n" "$RUNS" "$f.in" "$out.res" > "$out.times" 2> "$out.log"; then
    echo "- CRASH"; return
  fi
  local ok=OK
  futhark datacmp "$out.res" "$f.out" > "$out.cmp" 2>&1 || ok=FAIL
  rm -f "$out.res"
  echo "$(median_ms "$out.times") $ok"
}

printf 'd\tN\tsection\tsection_title\tcommit\tstep\tms\tcheck\n' > "$TSV"
for size in $SIZES; do
  d="${size%,*}" n="${size#*,}"
  f="data/d$d-N$n"
  [ -f "$f.in" ] && [ -f "$f.out" ] || ./gen-datasets.sh "$size"
  echo "d=$d N=$n" >&2

  read -r ms ok < <(run "$ORIG" "$d" "$n" "$f" original)
  printf '%s\t%s\t0\t%s\tbaseline\tCUDA baseline\t%s\t%s\n' \
    "$d" "$n" "${SECTION_TITLES[0]}" "$ms" "$ok" >> "$TSV"

  for step in "${STEPS[@]}"; do
    IFS='|' read -r section commit label <<< "$step"
    echo "  $commit $label" >&2
    read -r ms ok < <(run "$BUILD/$commit/attention" "$d" "$n" "$f" "$commit")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$d" "$n" "$section" "${SECTION_TITLES[$section]}" "$commit" "$label" "$ms" "$ok" >> "$TSV"
  done
done

# ---- Report ------------------------------------------------------------------
awk -F'\t' '
  function x(a, b) { return (a == "-" || b == "-" || b == 0) ? "-" : sprintf("%.2fx", a / b) }
  NR == 1 { next }
  {
    key = $1 "," $2
    if (!(key in nrows)) order[++nkeys] = key
    i = ++nrows[key]
    sec[key, i] = $3; com[key, i] = $5; lab[key, i] = $6; ms[key, i] = $7; chk[key, i] = $8
    if ($3 > 0) {
      title[$3] = $4; if ($3 > maxsec) maxsec = $3
      if ($7 != "-") last[key, $3] = $7
    }
  }
  END {
    for (k = 1; k <= nkeys; k++) {
      key = order[k]; split(key, dn, ",")
      printf "\n### d = %s, N = %s\n\n", dn[1], dn[2]
      print "| Sec | Step | Commit | ms | Check | vs previous | vs naive | vs baseline |"
      print "| --: | :-- | :-- | --: | :-: | --: | --: | --: |"
      base = naive = prev = "-"
      for (i = 1; i <= nrows[key]; i++) {
        m = ms[key, i]
        if (sec[key, i] == 0) {
          base = m
          printf "| - | %s | - | %s | %s | - | - | 1.00x |\n", lab[key, i], m, chk[key, i]
          continue
        }
        if (sec[key, i] == 1) naive = m
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", sec[key, i], lab[key, i], com[key, i], \
               m, chk[key, i], x(prev, m), x(naive, m), x(base, m)
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
    print "\nChecks: futhark datacmp against data/*.out. Medians over the runs of each version."
    print "Steps before 978f0ce are timed with gettimeofday + cudaDeviceSynchronize, later ones with CUDA events;"
    print "the 978f0ce row (same kernel as 1dbb008) shows the size of that difference."
  }
' "$TSV"
