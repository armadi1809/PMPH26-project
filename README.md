# FlashAttention on the A100

Course project for PMPH.
We implement FlashAttention-2, optimize it step by step on the CUDA cores, and
then move both matrix products to the TF32 tensor cores. We compare against the CUDA and
Futhark baselines of the CFAL paper.

All programs compute unscaled, non-causal, non-batched single-head attention in FP32,
`O = softmax(Q K^T) V`, with `Q`, `K`, `V` of size `N x d`. There is no `1/sqrt(d)`
scaling, mask or batch dimension.

## Repository layout

| Path | Contents |
| :-- | :-- |
| `attention-cuda/main.cu` | Our final CUDA core kernel (the last optimization step) |
| `attention-cuda/main_tensor.cu` | Our TF32 tensor core kernel (needs `sm_80`) |
| `original-cuda/` | CUDA baseline |
| `futhark/` | Futhark baseline, which also produces the reference outputs |
| `utils/utils.cuh` | Shared host code for reading inputs, writing outputs and error checks |
| `gen-datasets.sh` | Generates random inputs and reference outputs in `data/` |
| `compare.sh` | Checks and times the CUDA baseline, the tensor core kernel and Futhark |
| `steps.sh` | Builds, checks and times every optimization step from its commit |

## Requirements

An NVIDIA A100 (or another `sm_80` GPU), `nvcc`, and `futhark` on the `PATH`.
Futhark generates the datasets and checks every output with `futhark datacmp`.

## Running the benchmarks

```sh
DEVICE=0 ./compare.sh 64,16384 64,32768 128,8192 128,16384
DEVICE=0 ./steps.sh
```

Both scripts take dataset sizes as `d,N` arguments and generate missing datasets
themselves. `steps.sh` defaults to the four benchmark sizes above. Environment variables:
`DEVICE` (GPU index as in `nvidia-smi`, default 0), `RUNS` (timed runs per measurement,
default 10) and `ROUNDS` (default 3).

Every program does a warm-up run and is then timed `RUNS` times with `gettimeofday` (same as baseline)
around the kernel launch and `cudaDeviceSynchronize`. Each measurement is repeated in
`ROUNDS` rounds, and the scripts report the lowest of the round medians together with
the spread between the rounds. Check
`nvidia-smi` for other jobs before measuring, as it yields dubious numbers if other processes are using it as well.
`results/steps.tsv` and `results/steps-raw.tsv`.

## Optimization steps

Every step is a commit on this branch. `steps.sh` builds each of them directly from git,
without a checkout.

| # | Step | Commit |
| --: | :-- | :-- |
| 1 | Naive implementation | `0e114a0` |
| 2 | Switch loop order | `71f546a` |
| 3 | Pad K tile | `43e725d` |
| 4 | Register tiling of `Q K^T` | `006ad39` |
| 5 | Register tiling of `P V`, `O` in registers | `61fac95` |
| 6 | Online softmax in registers | `58ab269` |
| 7 | Reduce register pressure in `O` update | `05d7012` |
| 8 | Alias K and V in shared memory | `90f4de5` |
| 9 | Unroll fixed-trip-count loops | `d50b6c8` |
| 10 | Cap thread blocks per SM | `7fcb41d` |
| 11 | Keep k-loops rolled | `706b62f` |
| 12 | Swizzled shared-memory layouts | `4d0dd5d` |
| 13 | Prefetch V with `cp.async` | `b6218aa` |
| | TF32 tensor cores (first version, cleaned up in `44d942b`) | `e760a85` |

## Navigating the versions

To read a version or see what a step changed, no checkout is needed:

```sh
git show 4d0dd5d:attention-cuda/main.cu                    # the kernel of step 12
git diff 706b62f 4d0dd5d -- attention-cuda/main.cu         # what step 12 changed
```

To build and run a version, check out its commit, then return to the branch:

```sh
./gen-datasets.sh 64,16384                                 # once, if data/ is empty
git checkout 4d0dd5d
mkdir -p build
nvcc -O3 -arch=sm_80 -o build/attention attention-cuda/main.cu
build/attention 64 16384 10 data/d64-N16384.in build/out.res   # d N runs input output
futhark datacmp build/out.res data/d64-N16384.out              # compare to the reference
git checkout krebso/submission
```

Each program prints one runtime in seconds per line.
`d` must be 64 or 128, and `N` a multiple of `d`.
The tensor core kernel is built the same way from `attention-cuda/main_tensor.cu`.
Older commits also contain older versions of the scripts, so always measure with `steps.sh` and `compare.sh` from the latest commit.
