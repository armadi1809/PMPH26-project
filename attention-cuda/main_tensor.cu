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

constexpr int TILE = 64;

// Shape of the tensor core instruction mma.m16n8k8: D (M x N) += A (M x K) * B (K x N)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

template <int Br, int Bc, int D>
struct SmemLayout {
    // keys per tile actually used
    static constexpr int BC  = (D >= 128) ? 32 : Bc;
    // Row strides in floats. Rows are padded so that the fragment loads are free
    // of bank conflicts (D and BC are multiples of 32). With 4 extra floats, the
    // 8 rows of an ldmatrix matrix start at banks 0, 4, ..., 28 and cover all 32
    // banks once. With 8 extra floats, the plain loads of V (row t, column g)
    // hit bank 8t + g, which differs for all 32 lanes.
    static constexpr int Q_STRIDE = D + 4;
    static constexpr int K_STRIDE = D + 4;
    static constexpr int V_STRIDE = D + 8;
    static constexpr int P_STRIDE = BC + 4;
    // Tile sizes in floats
    static constexpr int Q_SIZE = Br * Q_STRIDE;
    static constexpr int K_SIZE = BC * K_STRIDE;
    static constexpr int V_SIZE = BC * V_STRIDE;
    static constexpr int P_SIZE = Br * P_STRIDE;
    static constexpr int TOTAL  = Q_SIZE + K_SIZE + V_SIZE + P_SIZE;
};

__device__ __forceinline__ uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

// c (16x8, FP32) += a (16x8, TF32) * b (8x8, TF32)
// Fragment layouts:
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

__device__ __forceinline__ void ldmatrix_x4_tf32(uint32_t (&r)[4], const float* p) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
    #pragma unroll
    for (int i = 0; i < 4; i++) r[i] = to_tf32(__uint_as_float(r[i]));
}

// Max and sum over the 4 threads of a quad, which together hold one row of S
__device__ __forceinline__ float quad_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}

__device__ __forceinline__ float quad_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
}

// Asynchronously copy a ROWS x D tile (row-major in global memory) into shared
// memory with row stride STRIDE, in 16-byte chunks. Consecutive threads copy
// consecutive chunks of a row, so the global reads are coalesced.
template <int ROWS, int STRIDE, int D, int NT>
__device__ __forceinline__ void load_tile_async(float* dst, const float* src, int tx) {
    constexpr int C4 = D / 4;
    constexpr int CHUNKS = ROWS * C4;
    static_assert(CHUNKS % NT == 0, "tile must divide evenly among threads");
    #pragma unroll
    for (int e = 0; e < CHUNKS / NT; e++) {
        const int f   = e * NT + tx;
        const int row = f / C4;
        const int c4  = f % C4;
        __pipeline_memcpy_async(&dst[row * STRIDE + c4 * 4], &src[row * D + c4 * 4], 16);
    }
}

float* allocate_host_memory(size_t cnt) {
    return (float*) calloc(cnt, sizeof(float));
}

template <int Br, int Bc, int d>
__global__ void __launch_bounds__((Br / MMA_M) * 32)
s2mm_kernel(int N, const float* Q, const float* K, const float* V, float* O)
{
    using L = SmemLayout<Br, Bc, d>;
    constexpr int BC  = L::BC;
    constexpr int NT  = (Br / MMA_M) * 32;       // one warp per 16 query rows: 128 threads
    constexpr int NTS = BC / MMA_N;              // 8-wide n-tiles of S per warp
    constexpr int NTO = d / MMA_N;               // 8-wide n-tiles of O per warp

    static_assert(Br == TILE, "query tile is fixed at 64 rows");
    static_assert(d % MMA_N == 0 && d % MMA_K == 0 && BC % MMA_N == 0 && BC % MMA_K == 0,
                  "tile sizes must be multiples of the mma shape");
    static_assert(NTS % 2 == 0, "K fragments are loaded two n-tiles at a time");

    extern __shared__ __align__(16) float smem[];
    float* Qs = smem;
    float* Ks = Qs + L::Q_SIZE;
    float* Vs = Ks + L::K_SIZE;
    float* Ps = Vs + L::V_SIZE;

    // Fragment coordinates (groupID and threadID_in_group in the PTX ISA): this
    // thread holds rows g and g + 8 of every S and O tile, at columns 2t and 2t + 1.
    const int tx       = threadIdx.x;
    const int lane     = tx % 32;
    const int g        = lane / 4;
    const int t        = lane % 4;
    const int warp_row = MMA_M * (tx / 32);      // this warp's first query row in the tile
    const int Tc       = N / BC;                 // number of K/V tiles

    // Shared-memory addresses of this thread's fragments, offset by k and n in the loops.
    // With ldmatrix, lanes 0-7, 8-15, 16-23 and 24-31 give the rows of the four matrices:
    //   Q, P (A): rows 0-7 and 8-15 of columns 0-3, then the same rows of columns 4-7
    //   K    (B): columns 0-3 and 4-7 of keys 0-7, then the same columns of keys 8-15
    const float* q_smem  = &Qs[(warp_row + lane % 16) * L::Q_STRIDE + 4 * (lane / 16)];
    const float* p_smem  = &Ps[(warp_row + lane % 16) * L::P_STRIDE + 4 * (lane / 16)];
    const float* k_smem  = &Ks[(8 * (lane / 16) + lane % 8) * L::K_STRIDE + 4 * (lane / 8 % 2)];
    const float* v_smem  = &Vs[t * L::V_STRIDE + g];                 // B[t][g] = V[t][g], plain loads
    float*       p_store = &Ps[(warp_row + g) * L::P_STRIDE + 2 * t];    // P in the layout of S

    // PROLOGUE
    // start copying Q_i and K_1, and initialize O and the running statistics
    load_tile_async<Br, L::Q_STRIDE, d, NT>(Qs, Q + (size_t)blockIdx.x * Br * d, tx);
    load_tile_async<BC, L::K_STRIDE, d, NT>(Ks, K, tx);
    __pipeline_commit();

    float o[NTO][4] = {};
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.f, 0.f};

    // MAIN LOOP
    // one K_j, V_j pair per iteration
    for (int j = 0; j < Tc; j++) {
        // my copies of K_j (and Q) have arrived
        __pipeline_wait_prior(0);                
        // everyone's have; everyone is done with V_{j-1} and P
        __syncthreads();                         

        // Start V_j load, it arrives while we compute Q K_j^T
        load_tile_async<BC, L::V_STRIDE, d, NT>(Vs, V + (size_t)j * BC * d, tx);
        __pipeline_commit();

        // S = Q K^T for this warp's part of the S tile
        float s[NTS][4] = {};
        #pragma unroll
        for (int k = 0; k < d; k += MMA_K) {
            // q fragment
            uint32_t q[4];
            ldmatrix_x4_tf32(q, q_smem + k);
            #pragma unroll
            // walking along keys, 2 tiles at a time
            // because one load covers 16 keys (16 / 8 = 2)
            for (int n = 0; n < NTS; n += 2) {
                // fragments of K^T for two n-tiles
                uint32_t k_T[4];                 
                ldmatrix_x4_tf32(k_T, k_smem + MMA_N * n * L::K_STRIDE + k);
                mma_tf32(s[n],     q, k_T[0], k_T[1]);
                mma_tf32(s[n + 1], q, k_T[2], k_T[3]);
            }
        }

        // Online softmax. Row g + 8h of S is s[n][2h] and s[n][2h + 1] for every n.
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float mx = -INFINITY;
            #pragma unroll
            for (int n = 0; n < NTS; n++) mx = fmaxf(mx, fmaxf(s[n][2 * h], s[n][2 * h + 1]));
            const float m_new = fmaxf(row_max[h], quad_max(mx));
            const float alpha = expf(row_max[h] - m_new);       // 0 on the first tile

            float sum = 0.f;
            #pragma unroll
            for (int n = 0; n < NTS; n++) {
                s[n][2 * h]     = expf(s[n][2 * h]     - m_new);
                s[n][2 * h + 1] = expf(s[n][2 * h + 1] - m_new);
                sum += s[n][2 * h] + s[n][2 * h + 1];
            }
            row_sum[h] = alpha * row_sum[h] + quad_sum(sum);
            row_max[h] = m_new;

            // Rescale the accumulator (normalization by the sum happens once, at the end)
            #pragma unroll
            for (int n = 0; n < NTO; n++) { o[n][2 * h] *= alpha;  o[n][2 * h + 1] *= alpha; }
        }

        // Publish this warp P rows for the P V product
        #pragma unroll
        for (int n = 0; n < NTS; n++) {
            *reinterpret_cast<float2*>(p_store + MMA_N * n)                  = make_float2(s[n][0], s[n][1]);
            *reinterpret_cast<float2*>(p_store + 8 * L::P_STRIDE + MMA_N * n) = make_float2(s[n][2], s[n][3]);
        }
        // my copies of V_j have arrived
        __pipeline_wait_prior(0);
        // P is visible
        __syncthreads();                         

        // Start loading K_{j+1}
        if (j + 1 < Tc)
            load_tile_async<BC, L::K_STRIDE, d, NT>(Ks, K + (size_t)(j + 1) * BC * d, tx);
        __pipeline_commit();

        // O += P V
        #pragma unroll
        for (int k = 0; k < BC; k += MMA_K) {
            // fragment of P
            uint32_t p[4];
            ldmatrix_x4_tf32(p, p_smem + k);
            #pragma unroll
            for (int n = 0; n < NTO; n++) {
                const float* v = v_smem + k * L::V_STRIDE + MMA_N * n;   // fragment of V (B operand)
                mma_tf32(o[n], p, to_tf32(v[0]), to_tf32(v[4 * L::V_STRIDE]));
            }
        }
    }

    // EPILOGUE
    // divide O by the running sums and write it straight from registers
    const float inv0 = 1.f / row_sum[0], inv1 = 1.f / row_sum[1];
    float* Ob = O + ((size_t)blockIdx.x * Br + warp_row + g) * d + 2 * t;
    #pragma unroll
    for (int n = 0; n < NTO; n++) {
        *reinterpret_cast<float2*>(Ob + MMA_N * n)         = make_float2(o[n][0] * inv0, o[n][1] * inv0);
        *reinterpret_cast<float2*>(Ob + 8 * d + MMA_N * n) = make_float2(o[n][2] * inv1, o[n][3] * inv1);
    }
}

template <int Br, int Bc, int d>
void setParamsAndCallKer(int N, float *Q_d, float *K_d, float *V_d, float *O_d) {
    constexpr int NT = (Br / MMA_M) * 32;
    dim3 block(NT);
    dim3 grid(N / Br);
    const size_t shmem_size = SmemLayout<Br, Bc, d>::TOTAL * sizeof(float);

    // One-time setup, done during the warm-up run so it stays out of the timed runs
    static bool configured = false;
    if (!configured) {
        gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, d>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
        gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, d>,
                  cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        int blocks_per_sm;
        gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm, s2mm_kernel<Br, Bc, d>, NT, shmem_size));
        fprintf(stderr, "d = %d: %d blocks per SM, %zu bytes of shared memory per block\n",
                d, blocks_per_sm, shmem_size);
        configured = true;
    }

    s2mm_kernel<Br, Bc, d><<<grid, block, shmem_size>>>(N, Q_d, K_d, V_d, O_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{
    if (d == 128) {
        setParamsAndCallKer<64, 64, 128>(N, Q_d, K_d, V_d, O_d);
    } else if (d == 64) {
        setParamsAndCallKer<64, 64, 64>(N, Q_d, K_d, V_d, O_d);
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
            4 * cnt * sizeof(float));

    struct timeval ts, te;
    double dur = 0.0;
    {
        float *Q_d, *K_d, *V_d, *O_d;

        gpuAssert(cudaSetDevice(0));

        gpuAssert(cudaMalloc((void**) &Q_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &O_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &V_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &K_d, cnt*sizeof(float)));

        gpuAssert(cudaMemcpy(Q_d, Q, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(V_d, V, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(K_d, K, cnt*sizeof(float), cudaMemcpyHostToDevice));

        fprintf(stderr, "Warming up...");
        s2mm(O_d, Q_d, K_d, V_d, N, d);
        gpuAssert(cudaDeviceSynchronize());
        fprintf(stderr, "done.\n");

        fprintf(stderr, "Running s2mm...");
        fprintf(stderr, "Printing runtimes to stdout\n");

        // Wall-clock timing, as on the aziz-improvements branch
        for (int it = 0; it < runs; it++) {
            gettimeofday(&ts, NULL);
            s2mm(O_d, Q_d, K_d, V_d, N, d);
            gpuAssert(cudaDeviceSynchronize());
            gettimeofday(&te, NULL);
            dur = (double)(te.tv_usec - ts.tv_usec) / 1e6 +
                  (double)(te.tv_sec - ts.tv_sec);
            printf("%f\n", dur);
        }

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
    }

    free(Q);
    free(K);
    free(V);
    free(O);

    return status;
}

