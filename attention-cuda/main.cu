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
#include "../utils/utils.cuh"

#define GPU_RUNS 50
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

// All tiles are 64 wide, and N must be a multiple of 64.
// This lets the kernel drop every bounds check.
constexpr int TILE = 64;

template <int Br, int Bc, int D>
struct SmemLayout {
    static constexpr int Q     = Br * D;   // Q tile, transposed: D rows of Br (swizzled)
    static constexpr int KV    = Bc * D;   // K tile, transposed: D rows of Bc (swizzled); then V: Bc rows of D
    static constexpr int S     = Bc * Br;  // P tile, transposed: Bc rows of Br (swizzled)
    static constexpr int TOTAL = Q + KV + S;
};

// Index into a transposed tile with W floats per row: group (x / 4) of row k
// is stored at position (x / 4) XOR (k mod W/4).
template <int W>
__device__ __forceinline__ int swz(int k, int x) {
    constexpr int G = W / 4;
    static_assert((G & (G - 1)) == 0, "groups per row must be a power of 2");
    return k * W + ((((x >> 2) ^ (k & (G - 1))) << 2) | (x & 3));
}

float* allocate_host_memory(size_t cnt) {
    return (float*) calloc(cnt, sizeof(float));
}

// m and l are unused (the statistics live in registers); they are kept so the
// kernel signature, and therefore its mangled name, stays the same.
template <int Br, int Bc, int TN, int d>
__global__ void __launch_bounds__((Br / TN) * (Bc / TN))
s2mm_kernel(int N, const float* Q, const float* K, const float* V, float* O, float* m, float* l)
{
    constexpr int NT  = (Br / TN) * (Bc / TN);   // threads per block: 256
    constexpr int TC  = Bc / TN;                 // thread-columns: 16
    constexpr int DCH = d / (TC * TN);           // output column chunks per thread: 1 or 2
    constexpr int KCH = TILE;                    // length of the fully unrolled Q K^T chunk

    constexpr int Q_LOADS  = Br * d / NT;        // scalar loads per thread for the Q tile
    constexpr int K_LOADS  = Bc * d / NT;        // scalar loads per thread for a K tile
    constexpr int V_LOADS4 = Bc * d / 4 / NT;    // float4 loads per thread for a V tile

    static_assert(TN == 4, "float4 accesses assume TN == 4");
    static_assert(Br == TILE && Bc == TILE, "tiles are fixed at 64");
    static_assert(d % KCH == 0, "d must be a multiple of 64");
    static_assert(d % (TC * TN) == 0, "d must be a multiple of TC * TN");
    static_assert((Br * d) % NT == 0, "Q tile must divide evenly among threads");
    static_assert((Bc * d) % (4 * NT) == 0, "K/V tiles must divide evenly among threads");
    static_assert(TC <= 32 && (TC & (TC - 1)) == 0, "row group must fit in a warp, power of 2");

    extern __shared__ __align__(16) float smem[];
    using L = SmemLayout<Br, Bc, d>;
    float* Qi  = smem;
    float* KVj = Qi  + L::Q;
    float* Sij = KVj + L::KV;

    const int tx = threadIdx.x;
    const int tr = tx / TC;
    const int tc = tx % TC;
    const int Tc = N / Bc;
    const int i  = blockIdx.x;

    // Q tile: element idx of the tile is row idx / d, column idx % d,
    // and it sits at global offset idx from the tile's start.
    const float* Qb = Q + (size_t)i * Br * d;
    #pragma unroll
    for (int e = 0; e < Q_LOADS; e++) {
        const int idx = e * NT + tx;
        Qi[swz<Br>(idx % d, idx / d)] = Qb[idx];
    }
    // (the barrier at the top of the j loop makes the Q tile visible)

    float o[DCH][TN][TN] = {};
    float m_reg[TN], l_reg[TN];
    #pragma unroll
    for (int r = 0; r < TN; r++) { m_reg[r] = -INFINITY; l_reg[r] = 0.f; }

    for (int j = 0; j < Tc; j++) {
        const float* Kb = K + (size_t)j * Bc * d;
        const float* Vb = V + (size_t)j * Bc * d;

        __syncthreads();                         // previous P V is done with KVj and Sij

        // K tile, stored transposed and swizzled
        #pragma unroll
        for (int e = 0; e < K_LOADS; e++) {
            const int idx = e * NT + tx;
            KVj[swz<Bc>(idx % d, idx / d)] = Kb[idx];
        }
        __syncthreads();                         // K tile complete

        // S = Q K^T: chunks of 64 steps, each fully unrolled
        float s[TN][TN] = {};
        #pragma unroll 1
        for (int k0 = 0; k0 < d; k0 += KCH) {
            #pragma unroll
            for (int kk = 0; kk < KCH; kk++) {
                const int k = k0 + kk;
                const float4 q4 = *reinterpret_cast<const float4*>(&Qi[swz<Br>(k, tr * TN)]);
                const float4 k4 = *reinterpret_cast<const float4*>(&KVj[swz<Bc>(k, tc * TN)]);
                const float q[TN]  = {q4.x, q4.y, q4.z, q4.w};
                const float kv[TN] = {k4.x, k4.y, k4.z, k4.w};
                #pragma unroll
                for (int r = 0; r < TN; r++)
                    #pragma unroll
                    for (int c = 0; c < TN; c++)
                        s[r][c] += q[r] * kv[c];
            }
        }

        // Online softmax in registers (no masking: N is a multiple of 64)
        float a[TN], b[TN];
        #pragma unroll
        for (int r = 0; r < TN; r++) {
            float mx = s[r][0];
            #pragma unroll
            for (int c = 1; c < TN; c++) mx = fmaxf(mx, s[r][c]);
            #pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

            float sum = 0.f;
            #pragma unroll
            for (int c = 0; c < TN; c++) { s[r][c] = expf(s[r][c] - mx); sum += s[r][c]; }
            #pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffff, sum, off);

            const float m_new = fmaxf(m_reg[r], mx);
            const float l_new = expf(m_reg[r] - m_new) * l_reg[r] + expf(mx - m_new) * sum;
            const float inv_l = 1.f / l_new;
            a[r] = l_reg[r] * expf(m_reg[r] - m_new) * inv_l;
            b[r] = expf(mx - m_new) * inv_l;
            m_reg[r] = m_new;
            l_reg[r] = l_new;
        }

        // Publish b * P, transposed and swizzled: one float4 per key
        #pragma unroll
        for (int c = 0; c < TN; c++) {
            const float4 p4 = make_float4(s[0][c] * b[0], s[1][c] * b[1],
                                          s[2][c] * b[2], s[3][c] * b[3]);
            *reinterpret_cast<float4*>(&Sij[swz<Br>(tc * TN + c, tr * TN)]) = p4;
        }
        __syncthreads();                         // everyone done with K; P visible

        // V tile, row-major, into the same buffer as K
        #pragma unroll
        for (int e = 0; e < V_LOADS4; e++) {
            const int f = e * NT + tx;
            reinterpret_cast<float4*>(KVj)[f] = reinterpret_cast<const float4*>(Vb)[f];
        }
        __syncthreads();                         // V tile complete

        // O = a * O + (b * P) V
        #pragma unroll
        for (int r = 0; r < TN; r++)
            #pragma unroll
            for (int q = 0; q < DCH; q++)
                #pragma unroll
                for (int cc = 0; cc < TN; cc++)
                    o[q][r][cc] *= a[r];

        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            const float4 p4 = *reinterpret_cast<const float4*>(&Sij[swz<Br>(c, tr * TN)]);
            const float p[TN] = {p4.x, p4.y, p4.z, p4.w};
            #pragma unroll
            for (int q = 0; q < DCH; q++) {
                const float4 v4 = *reinterpret_cast<const float4*>(&KVj[c * d + q * TC * TN + tc * TN]);
                const float v[TN] = {v4.x, v4.y, v4.z, v4.w};
                #pragma unroll
                for (int r = 0; r < TN; r++)
                    #pragma unroll
                    for (int cc = 0; cc < TN; cc++)
                        o[q][r][cc] += p[r] * v[cc];
            }
        }
    }

    // Write O straight from registers, one float4 per row and column chunk
    float* Ob = O + (size_t)i * Br * d;
    #pragma unroll
    for (int r = 0; r < TN; r++)
        #pragma unroll
        for (int q = 0; q < DCH; q++)
            *reinterpret_cast<float4*>(&Ob[(tr * TN + r) * d + q * TC * TN + tc * TN]) =
                make_float4(o[q][r][0], o[q][r][1], o[q][r][2], o[q][r][3]);
}

template <int Br, int Bc, int TN, int d>
void setParamsAndCallKer(int N, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d, float* l_d) {
    constexpr int NT = (Br / TN) * (Bc / TN);
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

