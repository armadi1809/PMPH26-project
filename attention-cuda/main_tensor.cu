#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "../utils/utils.cuh"

#define GPU_RUNS 50
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

// Tensor-core (TF32) version.
//
// Each warp owns 16 query rows and computes with the mma.sync m16n8k8 TF32
// instruction: D (16x8) += A (16x8) * B (8x8), FP32 accumulation.
// Q, K, V, P live in shared memory in plain row-major layout (padded rows);
// S, the softmax statistics and O live in registers.
// K and V are loaded with cp.async in a two-stage pipeline:
//   V_j arrives while Q K_j^T runs, K_{j+1} arrives while P V_j runs.
//
// N must be a multiple of 64. The Bc template argument is kept for the kernel
// name; for d = 128 the kernel internally uses 32 keys per tile so that
// 2 blocks fit on an SM.

constexpr int TILE = 64;

template <int Br, int Bc, int D>
struct SmemLayout {
    static constexpr int BC  = (D >= 128) ? 32 : Bc;  // keys per tile actually used
    // Row paddings chosen so every fragment load below is bank-conflict free
    static constexpr int LDQ = D + 4;
    static constexpr int LDK = D + 4;
    static constexpr int LDV = D + 8;
    static constexpr int LDP = BC + 4;
    static constexpr int Q   = Br * LDQ;
    static constexpr int K   = BC * LDK;
    static constexpr int V   = BC * LDV;
    static constexpr int P   = Br * LDP;
    static constexpr int TOTAL = Q + K + V + P;
};

// Round an FP32 value to TF32 (the tensor cores' input format)
__device__ __forceinline__ uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

// c (16x8, FP32) += a (16x8, TF32) * b (8x8, TF32), executed by the whole warp.
// Fragment layouts (g = lane / 4, t = lane % 4):
//   a[0] = A[g][t],   a[1] = A[g+8][t],   a[2] = A[g][t+4],   a[3] = A[g+8][t+4]
//   b0   = B[t][g],   b1   = B[t+4][g]
//   c[0] = C[g][2t],  c[1] = C[g][2t+1],  c[2] = C[g+8][2t],  c[3] = C[g+8][2t+1]
__device__ __forceinline__ void mma_tf32(float (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Asynchronously copy a ROWS x D tile (row-major in global memory) into shared
// memory with row stride LD, in 16-byte chunks. Consecutive threads copy
// consecutive chunks of a row, so the global reads are coalesced.
template <int ROWS, int LD, int D, int NT>
__device__ __forceinline__ void load_tile_async(float* dst, const float* src, int tx) {
    constexpr int C4 = D / 4;
    constexpr int CHUNKS = ROWS * C4;
    static_assert(CHUNKS % NT == 0, "tile must divide evenly among threads");
    #pragma unroll
    for (int e = 0; e < CHUNKS / NT; e++) {
        const int f   = e * NT + tx;
        const int row = f / C4;
        const int c4  = f % C4;
        __pipeline_memcpy_async(&dst[row * LD + c4 * 4], &src[row * D + c4 * 4], 16);
    }
}

float* allocate_host_memory(size_t cnt) {
    return (float*) calloc(cnt, sizeof(float));
}

// m and l are unused, and TN no longer matters; both are kept so the kernel
// signature, and therefore its mangled name, stays the same.
template <int Br, int Bc, int TN, int d>
__global__ void __launch_bounds__((Br / 16) * 32)
s2mm_kernel(int N, const float* Q, const float* K, const float* V, float* O, float* m, float* l)
{
    using L = SmemLayout<Br, Bc, d>;
    constexpr int BC  = L::BC;
    constexpr int NT  = (Br / 16) * 32;          // one warp per 16 query rows: 128 threads
    constexpr int NTS = BC / 8;                  // 8-wide n-tiles of S per warp
    constexpr int NTO = d / 8;                   // 8-wide n-tiles of O per warp

    static_assert(Br == TILE, "query tile is fixed at 64 rows");
    static_assert(d % 8 == 0 && BC % 8 == 0, "dimensions must be multiples of 8");

    extern __shared__ __align__(16) float smem[];
    float* Qs = smem;
    float* Ks = Qs + L::Q;
    float* Vs = Ks + L::K;
    float* Ps = Vs + L::V;

    const int tx    = threadIdx.x;
    const int warp  = tx >> 5;
    const int lane  = tx & 31;
    const int g     = lane >> 2;                 // row within the 8-row group
    const int t     = lane & 3;                  // thread within the quad
    const int prow  = warp * 16;                 // this warp's first query row in the tile
    const int Tc    = N / BC;

    // Prologue: start loading the Q tile and the first K tile
    load_tile_async<Br, L::LDQ, d, NT>(Qs, Q + (size_t)blockIdx.x * Br * d, tx);
    load_tile_async<BC, L::LDK, d, NT>(Ks, K, tx);
    __pipeline_commit();

    float o[NTO][4] = {};
    float m0 = -INFINITY, m1 = -INFINITY;        // running max for rows g and g+8
    float l0 = 0.f, l1 = 0.f;                    // running sum for rows g and g+8

    for (int j = 0; j < Tc; j++) {
        __pipeline_wait_prior(0);                // my copies of K_j (and Q) have arrived
        __syncthreads();                         // everyone's have; everyone is done with V_{j-1} and P

        // Start V_j; it arrives while Q K_j^T runs
        load_tile_async<BC, L::LDV, d, NT>(Vs, V + (size_t)j * BC * d, tx);
        __pipeline_commit();

        // S = Q K^T for this warp's 16 rows and all BC keys of the tile
        float s[NTS][4] = {};
        #pragma unroll
        for (int k0 = 0; k0 < d; k0 += 8) {
            const float* qp = &Qs[(prow + g) * L::LDQ + k0 + t];
            const uint32_t a[4] = { to_tf32(qp[0]),              to_tf32(qp[8 * L::LDQ]),
                                    to_tf32(qp[4]),              to_tf32(qp[8 * L::LDQ + 4]) };
            #pragma unroll
            for (int n = 0; n < NTS; n++) {
                const float* kp = &Ks[(n * 8 + g) * L::LDK + k0 + t];   // B[k][n] = K[n][k]
                mma_tf32(s[n], a, to_tf32(kp[0]), to_tf32(kp[4]));
            }
        }

        // Online softmax. This thread holds rows g (s[n][0..1]) and g+8 (s[n][2..3]);
        // each row is spread over the 4 threads of a quad.
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int n = 0; n < NTS; n++) {
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = expf(m0 - mn0),  al1 = expf(m1 - mn1);   // 0 on the first tile

        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int n = 0; n < NTS; n++) {
            s[n][0] = expf(s[n][0] - mn0);  s[n][1] = expf(s[n][1] - mn0);
            s[n][2] = expf(s[n][2] - mn1);  s[n][3] = expf(s[n][3] - mn1);
            sum0 += s[n][0] + s[n][1];
            sum1 += s[n][2] + s[n][3];
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, off);
        }
        l0 = al0 * l0 + sum0;  l1 = al1 * l1 + sum1;
        m0 = mn0;              m1 = mn1;

        // Rescale the accumulator (normalization by l happens once, at the end)
        #pragma unroll
        for (int n = 0; n < NTO; n++) {
            o[n][0] *= al0;  o[n][1] *= al0;
            o[n][2] *= al1;  o[n][3] *= al1;
        }

        // Publish P (this warp's rows only) for the P V product
        #pragma unroll
        for (int n = 0; n < NTS; n++) {
            *reinterpret_cast<float2*>(&Ps[(prow + g)     * L::LDP + n * 8 + 2 * t]) = make_float2(s[n][0], s[n][1]);
            *reinterpret_cast<float2*>(&Ps[(prow + g + 8) * L::LDP + n * 8 + 2 * t]) = make_float2(s[n][2], s[n][3]);
        }

        __pipeline_wait_prior(0);                // my copies of V_j have arrived
        __syncthreads();                         // everyone's have; P visible; everyone done with K_j

        // Start K_{j+1}; it arrives while P V_j runs
        if (j + 1 < Tc)
            load_tile_async<BC, L::LDK, d, NT>(Ks, K + (size_t)(j + 1) * BC * d, tx);
        __pipeline_commit();

        // O += P V
        #pragma unroll
        for (int kk = 0; kk < BC; kk += 8) {
            const float* pp = &Ps[(prow + g) * L::LDP + kk + t];
            const uint32_t a[4] = { to_tf32(pp[0]),              to_tf32(pp[8 * L::LDP]),
                                    to_tf32(pp[4]),              to_tf32(pp[8 * L::LDP + 4]) };
            #pragma unroll
            for (int n = 0; n < NTO; n++) {
                const float* vp = &Vs[(kk + t) * L::LDV + n * 8 + g];  // B[k][n] = V[k][n]
                mma_tf32(o[n], a, to_tf32(vp[0]), to_tf32(vp[4 * L::LDV]));
            }
        }
    }

    // Normalize and write O: this thread holds rows g and g+8, columns n*8 + 2t (+1)
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
    float* Ob = O + ((size_t)blockIdx.x * Br + prow) * d;
    #pragma unroll
    for (int n = 0; n < NTO; n++) {
        *reinterpret_cast<float2*>(&Ob[g * d       + n * 8 + 2 * t]) = make_float2(o[n][0] * inv0, o[n][1] * inv0);
        *reinterpret_cast<float2*>(&Ob[(g + 8) * d + n * 8 + 2 * t]) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
    }
}

template <int Br, int Bc, int TN, int d>
void setParamsAndCallKer(int N, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d, float* l_d) {
    constexpr int NT = (Br / 16) * 32;
    dim3 block(NT);
    dim3 grid(N / Br);
    const size_t shmem_size = SmemLayout<Br, Bc, d>::TOTAL * sizeof(float);

    // One-time setup, done during the warm-up run so it stays out of the timed runs
    static bool configured = false;
    if (!configured) {
        gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, TN, d>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
        gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, TN, d>,
                  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        int blocks_per_sm;
        gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm, s2mm_kernel<Br, Bc, TN, d>, NT, shmem_size));
        fprintf(stderr, "d = %d: %d blocks per SM, %zu bytes of shared memory per block\n",
                d, blocks_per_sm, shmem_size);
        configured = true;
    }

    s2mm_kernel<Br, Bc, TN, d><<<grid, block, shmem_size>>>(N, Q_d, K_d, V_d, O_d, m_d, l_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{
    if (d == 128) {
        setParamsAndCallKer<64, 64, 4, 128>(N, Q_d, K_d, V_d, O_d, m_d, l_d);
    } else if (d == 64) {
        setParamsAndCallKer<64, 64, 4, 64>(N, Q_d, K_d, V_d, O_d, m_d, l_d);
    } else {
        printf("Unsupported dataset, exiting!");
        exit(1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int status = EXIT_SUCCESS;
    if (argc != 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N runs in out Read Q, K, V from Futhark binary in, write O to out\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N, d, runs = GPU_RUNS;
    if (!parse_positive_int(argv[1], &d) ||
        !parse_positive_int(argv[2], &N) ||
        (argc >= 4 && !parse_positive_int(argv[3], &runs))) {
        fprintf(stderr, "d, N, and runs must be positive integers no greater than %d.\n", INT_MAX);
        return EXIT_FAILURE;
    }

    if (d != 64 && d != 128) {
        fprintf(stderr, "d must be 64 or 128.\n");
        return EXIT_FAILURE;
    }

    if (N % TILE != 0) {
        fprintf(stderr, "N must be a multiple of %d.\n", TILE);
        return EXIT_FAILURE;
    }

    if (N % d != 0) {
        fprintf(stderr, "d must divide N\n");
        return EXIT_FAILURE;
    }

    size_t cnt = (size_t)N * (size_t)d;

    fprintf(stderr, "\nN=%d, d=%d\n", N, d);
    fprintf(stderr, "Initializing data...");

    float * Q = allocate_host_memory(cnt);
    float * K = allocate_host_memory(cnt);
    float * V = allocate_host_memory(cnt);
    float * O = allocate_host_memory(cnt);

    if (Q == NULL || K == NULL || V == NULL || O == NULL) {
        fprintf(stderr, "Failed to allocate host memory");
        return EXIT_FAILURE;
    }

    const char *in_path = argv[4];
    const char *out_path = argv[5];

    if (!read_inputs(in_path, Q, K, V, cnt)) {
        fprintf(stderr, "Failed to read %zu floats per matrix from %s.\n", cnt, in_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "done.\n");
    fprintf(stderr, "Device bytes to allocate: %zu\n",
            4 * cnt * sizeof(float) + 2 * (size_t)N * sizeof(float));

    double dur = 0.0;
    {
        float *Q_d, *K_d, *V_d, *O_d, *m_d, *l_d;

        gpuAssert(cudaSetDevice(0));

        gpuAssert(cudaMalloc((void**) &Q_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &O_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &V_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &K_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &m_d, (size_t)N*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &l_d, (size_t)N*sizeof(float)));

        gpuAssert(cudaMemcpy(Q_d, Q, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(V_d, V, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(K_d, K, cnt*sizeof(float), cudaMemcpyHostToDevice));

        fprintf(stderr, "Warming up...");
        s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);
        gpuAssert(cudaDeviceSynchronize());
        fprintf(stderr, "done.\n");

        fprintf(stderr, "Running s2mm...");
        fprintf(stderr, "Printing runtimes to stdout\n");

        cudaEvent_t start, stop;
        gpuAssert(cudaEventCreate(&start));
        gpuAssert(cudaEventCreate(&stop));

        for (int it = 0; it < runs; it++) {
            gpuAssert(cudaEventRecord(start));
            s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);
            gpuAssert(cudaEventRecord(stop));
            gpuAssert(cudaEventSynchronize(stop));

            float ms;
            gpuAssert(cudaEventElapsedTime(&ms, start, stop));
            dur = ms / 1e3;
            printf("%f\n", dur);
        }

        gpuAssert(cudaEventDestroy(start));
        gpuAssert(cudaEventDestroy(stop));

        gpuAssert(cudaMemcpy(O, O_d, cnt*sizeof(float), cudaMemcpyDeviceToHost));
        gpuAssert(cudaDeviceSynchronize());

        if (!write_output(out_path, O, N, d)) {
            fprintf(stderr, "Failed to write output to %s.\n", out_path);
            status = EXIT_FAILURE;
        }

        /* QK^t is 2N^2d flops, so is PV. */
        fprintf(stderr,
                "Compute rate: %lf Gflops/s, runtime: %lf ms\n",
                4.0 * d * N * N / dur / 1e9, dur * 1e3);

        gpuAssert(cudaFree(Q_d));
        gpuAssert(cudaFree(K_d));
        gpuAssert(cudaFree(V_d));
        gpuAssert(cudaFree(O_d));
        gpuAssert(cudaFree(m_d));
        gpuAssert(cudaFree(l_d));
    }

    free(Q);
    free(K);
    free(V);
    free(O);

    return status;
}

