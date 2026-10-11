# PMPH26 FlashAttention project

Programming Massively Parallel Hardware course project. The goal is to develop
and optimize our own CUDA FlashAttention implementation and compare it against some baselines like the cuda and futhark ones here.

## Quick start on the GPU machine

```sh
cd PMPH26-project
make -C cuda compile
make -C cuda run D=64 N=1024
make -C cuda test
make -C cuda bench

make -C futhark compile
make -C futhark run D=64 N=1024
make -C futhark test
make -C futhark bench
```

`compile` is the default target in each subfolder. `run` is a small standalone
execution; `bench` uses the following four baseline workloads:

| Head dimension `d` | Sequence length `N` |
| ------------------ | ------------------- |
| 64                 | 16384               |
| 64                 | 32768               |
| 128                | 8192                |
| 128                | 16384               |

Those full benchmarks are intentionally large. Start with `run` or `test`.
For a CPU-only Futhark use:

```sh
make -C futhark check
make -C futhark test BACKEND=c
make -C futhark run BACKEND=multicore D=64 N=1024 THREADS=4
make -C futhark bench_smoke BACKEND=c RUNS=1
```

## What is being compared

Both baselines calculate **unscaled, non-causal, single-head attention**:

```text
O = softmax(Q K^T) V
```

There is no `1/sqrt(d)` scaling, causal mask, dropout, or batch dimension.
The supplied workloads fill `Q`, `K`, and `V` with ones; the expected output is
an `N x d` matrix of ones. Our optimized CUDA implementation must match this
operation and these sizes for comparable timings.

- The CUDA baseline emits one runtime in **seconds** per line, with 50 measured
  runs by default after one warmup. These are host wall-clock measurements
  around a kernel launch and device synchronization, excluding input generation
  and host/device transfers. `bench` saves them under [cuda/results/](cuda/results/).
- Futhark uses default parameters and `futhark bench`, including its own warmup
  and statistical methodology. Its JSON `runtimes` are in **microseconds**. The default benchmark
  selects `main64` and `main128`, which return the attention output, rather than
  timing the extra L2 reduction in `validate`. Results go under
  [futhark/results/](futhark/results/).
  Use `make -C futhark bench ENTRIES=validate` to include the validation
  reduction in the timing instead.
- The Futhark baseline computes a `d x N` score block and uses a chunked online
  softmax before multiplying by `V`. It is not the same tiled/fused kernel as
  the CUDA baseline, so their intermediate memory use and execution strategy
  differ.
