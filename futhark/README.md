# Futhark baseline

Runs and benchmarks use Futhark's default parameters.

From the project root:

```sh
cd futhark
make run D=64 N=1024           # Build and run; print L2 deviation (~0f32)
make test                     # Small all-ones checks
make bench                    # Large benchmarks; save JSON in results/
```

Without CUDA for testing:

```sh
make run BACKEND=c D=64 N=1024
make test BACKEND=c
```

`D` must be 64 or 128; `N` must be a positive multiple of 1024.
Inputs are generated automatically with all values set to one.

GPU selection defaults to `DEVICE=0`. Under Slurm, append
`DEVICE="$CUDA_VISIBLE_DEVICES"` to preserve the assigned GPU.
