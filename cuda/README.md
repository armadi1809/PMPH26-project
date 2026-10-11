# CUDA baseline

From the project root:

```sh
cd cuda
make run D=64 N=1024 RUNS=50    # Build and run
make test                     # Small all-ones checks
make bench                    # Large benchmarks; save timings in results/
```

`D` must be 64 or 128; `N` must be a positive multiple of `D`; `RUNS` must
be positive. Runs print timings in seconds using all-ones inputs.

GPU selection defaults to `DEVICE=0`. Under Slurm, append
`DEVICE="$CUDA_VISIBLE_DEVICES"` to preserve the assigned GPU.
