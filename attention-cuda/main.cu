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
#include "../utils/utils.cuh"

#define GPU_RUNS 50

float *allocate_host_memory(size_t cnt)
{
    float *ptr;
    if ((ptr = (float *)calloc(cnt, sizeof(float))) == NULL)
    {
        return NULL;
    }

    return ptr;
}

__device__ __forceinline__ float float4_get(const float4 &v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y
                      : i == 2   ? v.z
                                 : v.w;
}

// FlashAttention-2 style kernel with register tiling.
//   - A block has WARPS warps and covers BR = WARPS * R consecutive rows of O,
//     so every K/V chunk loaded into shared memory is shared by BR rows.
//   - A chunk has BC = 32 * KPL keys: every lane handles KPL keys of it
//     (lane j handles keys j, j+32, ...).
//   - Scores: lane j computes the scores of its KPL keys for all R rows of its warp.
//     Each K value read from shared memory is reused for R rows, and each Q value
//     (read as float4 broadcasts) is reused for KPL keys.
//   - Softmax per row with warp shuffles; m, l and the output stay in registers.
//     With KPL > 1, the shuffles, the rescaling and the barriers are paid once per
//     BC keys instead of once per 32 keys.
//   - P.V: P goes through a small per-warp shared buffer so it can be read as
//     float4 broadcasts; each V value read is reused for R rows.
//   - O is divided by l only once at the end.
// Assumes N % (32 * KPL) == 0 and N % (WARPS * R) == 0.
template <int D, int WARPS, int R, int KPL>
__global__ void __launch_bounds__(WARPS * 32)
    fa_tiled(int N, const float *__restrict__ Q, const float *__restrict__ K,
             const float *__restrict__ V, float *__restrict__ O)
{
    const int BC = 32 * KPL;   // keys per chunk
    const int BR = WARPS * R;  // query rows per block
    const int DPL = D / 32;    // output columns per lane
    const int KS = D + 4;      // padded row stride of sK: float4 reads without bank conflicts
    const int NT = WARPS * 32; // threads per block

    extern __shared__ __align__(16) float smem[];
    float *sK = smem;         // [BC][KS]
    float *sV = sK + BC * KS; // [BC][D]
    float *sQ = sV + BC * D;  // [BR][D]
    float *sP = sQ + BR * D;  // [WARPS][R][BC]

    // identify own lane, warp, starting row
    // as well as it's part of shared Q and P
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * BR + warp * R; // first row of this warp
    const float *myQ = sQ + warp * R * D;
    float *myP = sP + warp * R * BC;

    // The block's BR rows of Q are contiguous in global memory.
    // make threads collaborate to move the BR rows for the block
    // from glbal to shared memory
    {
        const float4 *src = reinterpret_cast<const float4 *>(Q + (size_t)blockIdx.x * BR * D);
        float4 *dst = reinterpret_cast<float4 *>(sQ);
        for (int i = threadIdx.x; i < BR * D / 4; i += NT)
            dst[i] = src[i];
    }

    // initialize running values
    // each thread owns a running max, l for the rows (R) is is responsible for

    float m[R], l[R], acc[R][DPL];
#pragma unroll
    for (int r = 0; r < R; r++)
    {
        m[r] = -INFINITY;
        l[r] = 0.f;
#pragma unroll
        for (int t = 0; t < DPL; t++)
            acc[r][t] = 0.f;
    }

    // chunks loading
    for (int kb = 0; kb < N; kb += BC)
    {
        __syncthreads(); // previous chunk (sK, sV, sP) no longer in use
        {
            const float4 *srcK = reinterpret_cast<const float4 *>(K + (size_t)kb * D);
            const float4 *srcV = reinterpret_cast<const float4 *>(V + (size_t)kb * D);
            for (int i = threadIdx.x; i < BC * D / 4; i += NT)
            {
                const int r = (4 * i) / D, c = (4 * i) % D;
                *reinterpret_cast<float4 *>(&sK[r * KS + c]) = srcK[i];
                *reinterpret_cast<float4 *>(&sV[r * D + c]) = srcV[i];
            }
        }
        __syncthreads();

        // Scores: s[r][k] = Q[row0 + r] . K[kb + lane + 32 * k]
        float s[R][KPL];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int k = 0; k < KPL; k++)
                s[r][k] = 0.f;
#pragma unroll 4
        // we use float4 here that's why we stride by 4
        for (int c = 0; c < D; c += 4)
        {
            float4 k4[KPL];
#pragma unroll
            for (int k = 0; k < KPL; k++)
                k4[k] = *reinterpret_cast<const float4 *>(&sK[(lane + 32 * k) * KS + c]);
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                const float4 q4 = *reinterpret_cast<const float4 *>(&myQ[r * D + c]);
#pragma unroll
                for (int k = 0; k < KPL; k++)
                {
                    s[r][k] += q4.x * k4[k].x;
                    s[r][k] += q4.y * k4[k].y;
                    s[r][k] += q4.z * k4[k].z;
                    s[r][k] += q4.w * k4[k].w;
                }
            }
        }

        // Online softmax, one row at a time, over the BC scores held by the warp
        float alpha[R];
#pragma unroll
        for (int r = 0; r < R; r++)
        {
            // max over my KPL scores, then over the 32 lanes
            float mt = s[r][0];
#pragma unroll
            for (int k = 1; k < KPL; k++)
                mt = fmaxf(mt, s[r][k]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float m_new = fmaxf(m[r], mt);

            // my KPL weights and their sum, then the sum over the 32 lanes
            float ps = 0.f;
#pragma unroll
            for (int k = 0; k < KPL; k++)
            {
                const float p = expf(s[r][k] - m_new);
                ps += p;
                myP[r * BC + lane + 32 * k] = p;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ps += __shfl_xor_sync(0xffffffff, ps, off);

            alpha[r] = expf(m[r] - m_new); // 0 on the first chunk (m = -inf)
            l[r] = l[r] * alpha[r] + ps;
            m[r] = m_new;
        }
        __syncwarp(); // this warp's P is complete

// acc = alpha * acc + P V   (lane owns columns lane, lane+32, ...)
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int t = 0; t < DPL; t++)
                acc[r][t] *= alpha[r];

#pragma unroll 2
        for (int j = 0; j < BC; j += 4)
        {
            float4 p4[R];
#pragma unroll
            for (int r = 0; r < R; r++)
                p4[r] = *reinterpret_cast<const float4 *>(&myP[r * BC + j]);
#pragma unroll
            for (int jj = 0; jj < 4; jj++)
            {
                float v[DPL];
#pragma unroll
                for (int t = 0; t < DPL; t++)
                    v[t] = sV[(j + jj) * D + lane + 32 * t];
#pragma unroll
                for (int r = 0; r < R; r++)
                {
                    const float pr = float4_get(p4[r], jj);
#pragma unroll
                    for (int t = 0; t < DPL; t++)
                        acc[r][t] += pr * v[t];
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < DPL; t++)
            O[(size_t)(row0 + r) * D + lane + 32 * t] = acc[r][t] / l[r];
}

template <int D, int WARPS, int R, int KPL>
void launch_fa_tiled(int N, const float *Q_d, const float *K_d, const float *V_d, float *O_d)
{
    constexpr int BC = 32 * KPL;
    constexpr int BR = WARPS * R;
    assert(N % BC == 0 && N % BR == 0);

    const size_t shmem_size = (BC * (D + 4) + // sK (padded)
                               BC * D +       // sV
                               BR * D +       // sQ
                               BR * BC        // sP
                               ) *
                              sizeof(float);

    // Needed when the kernel uses more than 48 KB of shared memory; done once per instantiation.
    static bool configured = false;
    if (!configured)
    {
        gpuAssert(cudaFuncSetAttribute(fa_tiled<D, WARPS, R, KPL>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
        configured = true;
    }

    fa_tiled<D, WARPS, R, KPL><<<N / BR, WARPS * 32, shmem_size>>>(N, Q_d, K_d, V_d, O_d);
    gpuAssert(cudaPeekAtLastError());
}

// Tile sizes follow the FlashAttention paper: Bc = M / (4d) and Br = min(M / (4d), d),
// where M is the shared-memory budget per block (sQ, sK, sV and sP are ~4 tiles of B x d).
// M = 16384 floats (64 KB) lets 2 blocks run per SM on an A100 (164 KB shared memory).
// Rows per warp is fixed; keys per lane and warps per block follow from Bc and Br.
constexpr int SMEM_BUDGET_FLOATS = 16384;
constexpr int ROWS_PER_WARP = 8;

template <int D>
void launch_attention(int N, const float *Q_d, const float *K_d, const float *V_d, float *O_d)
{
    constexpr int BC = SMEM_BUDGET_FLOATS / (4 * D);
    constexpr int BR = BC < D ? BC : D;
    static_assert(BC % 32 == 0 && BR % ROWS_PER_WARP == 0,
                  "tile sizes must fit the warp layout (Bc multiple of 32, Br multiple of ROWS_PER_WARP)");
    launch_fa_tiled<D, BR / ROWS_PER_WARP, ROWS_PER_WARP, BC / 32>(N, Q_d, K_d, V_d, O_d);
}

// m_d and l_d are no longer needed (m and l live in registers); kept so main() is unchanged.
__host__ int
s2mm(float *m_d, float *l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{
    if (d == 128)
    {
        launch_attention<128>(N, Q_d, K_d, V_d, O_d);
    }
    else if (d == 64)
    {
        launch_attention<64>(N, Q_d, K_d, V_d, O_d);
    }
    else
    {
        printf("Unsupported dataset, exiting!");
        exit(1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int status = EXIT_SUCCESS;
    if (argc != 6)
    {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N runs in_path out_path Compute with matrices filled with ones\n", argv[0]);
        fprintf(stderr, "  %s d N runs in out Read Q, K, V from Futhark binary in, write O to out\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N, d, runs = GPU_RUNS;
    if (!parse_positive_int(argv[1], &d) ||
        !parse_positive_int(argv[2], &N) ||
        (argc >= 4 && !parse_positive_int(argv[3], &runs)))
    {
        fprintf(stderr, "d, N, and runs must be positive integers no greater than %d.\n", INT_MAX);
        return EXIT_FAILURE;
    }

    if (d != 64 && d != 128)
    {
        fprintf(stderr, "d must be 64 or 128.\n");
        return EXIT_FAILURE;
    }

    if (N % d != 0)
    {
        fprintf(stderr, "d must divide N\n");
        return EXIT_FAILURE;
    }

    size_t cnt = (size_t)N * (size_t)d;

    fprintf(stderr, "\nN=%d, d=%d\n", N, d);

    fprintf(stderr, "Initializing data...");

    float *Q = allocate_host_memory(cnt);
    float *K = allocate_host_memory(cnt);
    float *V = allocate_host_memory(cnt);
    float *O = allocate_host_memory(cnt);

    if (Q == NULL || K == NULL || V == NULL || O == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory");
        return EXIT_FAILURE;
    }

    const char *in_path = argv[4];
    const char *out_path = argv[5];

    if (!read_inputs(in_path, Q, K, V, cnt))
    {
        fprintf(stderr, "Failed to read %zu floats per matrix from %s.\n", cnt, in_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "done.\n");

    fprintf(stderr, "Device bytes to allocate: %zu\n",
            4 * cnt * sizeof(float) + 2 * (size_t)N * sizeof(float));

    struct timeval ts, te;
    double dur;
    {
        float *Q_d, *K_d, *V_d, *O_d, *m_d, *l_d;

        gpuAssert(cudaSetDevice(0));

        // allocate memory on device
        gpuAssert(cudaMalloc((void **)&Q_d, cnt * sizeof(float)));
        gpuAssert(cudaMalloc((void **)&O_d, cnt * sizeof(float)));
        gpuAssert(cudaMalloc((void **)&V_d, cnt * sizeof(float)));
        gpuAssert(cudaMalloc((void **)&K_d, cnt * sizeof(float)));
        gpuAssert(cudaMalloc((void **)&m_d, (size_t)N * sizeof(float)));
        gpuAssert(cudaMalloc((void **)&l_d, (size_t)N * sizeof(float)));

        // copy host memory to device
        gpuAssert(cudaMemcpy(Q_d, Q, cnt * sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(V_d, V, cnt * sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(K_d, K, cnt * sizeof(float), cudaMemcpyHostToDevice));

        fprintf(stderr, "Warming up...");

        s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);

        gpuAssert(cudaDeviceSynchronize());

        fprintf(stderr, "done.\n");
        fprintf(stderr, "Running s2mm...");

        fprintf(stderr, "Printing runtimes to stdout\n");

        for (int i = 0; i < runs; i++)
        {
            gettimeofday(&ts, NULL);
            s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);
            gpuAssert(cudaDeviceSynchronize());
            gettimeofday(&te, NULL);
            dur = (double)(te.tv_usec - ts.tv_usec) / 1e6 +
                  (double)(te.tv_sec - ts.tv_sec);
            printf("%f\n", dur);
        }

        gpuAssert(cudaMemcpy(O, O_d, cnt * sizeof(float), cudaMemcpyDeviceToHost));
        gpuAssert(cudaDeviceSynchronize());

        if (!write_output(out_path, O, N, d))
        {
            fprintf(stderr, "Failed to write output to %s.\n", out_path);
            status = EXIT_FAILURE;
        }

        /* QK^t is 2N^2d flops, so is PV. softmax(S) (row-wise)
         * exp(S[i]) / sum_j exp(P[i, j] - max(P[i]))
         * is N * (N + 4N) = 5 N^2 flops, but exp is more expensive. */
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
