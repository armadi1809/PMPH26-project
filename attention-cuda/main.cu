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

template <int Br, int Bc, int D>
struct SmemLayout {
    static constexpr int Q     = Br * D;
    static constexpr int KV    = Bc * (D + 1);     // K tile (padded); V tile reuses it
    static constexpr int S     = Br * (Bc + 1);
    static constexpr int TOTAL = Q + KV + S;
};

float* allocate_host_memory(size_t cnt) {
    float * ptr;
    if ((ptr = (float*) calloc(cnt, sizeof(float))) == NULL) {
        return NULL;
    }

    return ptr;
}


template <int Br, int Bc, int TN, int d>
__global__ void __launch_bounds__((Br / TN) * (Bc / TN)) s2mm_kernel(int N, const float* Q, const float* K, const float* V, float* O, float *m, float*l) {
    
    constexpr int NT = (Br / TN) * (Bc / TN);
    constexpr int S_STRIDE = Bc + 1;
    constexpr int loads_KV = CEIL_DIV(Bc * d, NT);

    int tx = threadIdx.x;

    int Tc = CEIL_DIV(N, Bc);

    int qkv_off = 0;

    extern __shared__ float smem[];
    using L = SmemLayout<Br, Bc, d>;
    float* Qi  = smem;
    float* KVj = Qi  + L::Q;
    float* Sij = KVj + L::KV;

    int i = blockIdx.x;

    // load Q_i
    constexpr int elem_loads_per_thread_Q = CEIL_DIV(Br * d, NT);

    #pragma unroll
    for (int e = 0; e < elem_loads_per_thread_Q; e++) {
        int idx = e * (NT) + tx;
        if (idx < Br * d) {
            int row = idx / d;
            int col = idx % d;
            if (i * Br + row < N) {
                Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
            }
        }
    }

    // set O, m, l
    const int num_threads = blockDim.x;

    __syncthreads();

    const int tr = tx / (Bc / TN);
    const int tc = tx % (Bc / TN);

    constexpr int TC  = Bc / TN;          // thread-columns: 16
    constexpr int DCH = d / (TC * TN);    // column chunks per thread: 1 (d=64) or 2 (d=128)
    static_assert(d % (TC * TN) == 0, "d must be a multiple of TC * TN");

    float o[DCH][TN][TN] = {0.f};         // replaces Oi in shared memory
    
    static_assert(TC <= 32 && (TC & (TC - 1)) == 0, "row group must fit in a warp, power of 2");

    float m_reg[TN], l_reg[TN];
    #pragma unroll
    for (int r = 0; r < TN; r++) { m_reg[r] = -INFINITY; l_reg[r] = 0.f; }


    for (int j = 0; j < Tc; j++) {
        __syncthreads();
        
        #pragma unroll
        for (int e = 0; e < loads_KV; e++) {
            int idx = e * NT + tx;
            if (idx < Bc * d) {
                int row = idx / d, col = idx % d;
                if (j * Bc + row < N)
                    KVj[row * (d + 1) + col] = K[qkv_off + (j * Bc + row) * d + col];
            }
        }

        __syncthreads();

        float s[TN][TN] = {0.f};
        #pragma unroll
        for (int k = 0; k < d; k++) {
            float q[TN], kv[TN];
            #pragma unroll
            for (int r = 0; r < TN; r++) q[r]  = Qi[(tr * TN + r) * d + k];
            #pragma unroll
            for (int c = 0; c < TN; c++) kv[c] = KVj[(tc * TN + c) * (d + 1) + k];
            #pragma unroll
            for (int r = 0; r < TN; r++)
                #pragma unroll
                for (int c = 0; c < TN; c++)
                    s[r][c] += q[r] * kv[c];
        }

        float a[TN];
        #pragma unroll
        for (int r = 0; r < TN; r++) {
            // mask out-of-range keys
            #pragma unroll
            for (int c = 0; c < TN; c++)
                if (j * Bc + tc * TN + c >= N) s[r][c] = -INFINITY;

            // row max: own 4 values, then butterfly across the 16 lanes
            float mx = s[r][0];
            #pragma unroll
            for (int c = 1; c < TN; c++) mx = fmaxf(mx, s[r][c]);
            #pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

            // exponentials and row sum, same pattern
            float sum = 0.f;
            #pragma unroll
            for (int c = 0; c < TN; c++) { s[r][c] = expf(s[r][c] - mx); sum += s[r][c]; }
            #pragma unroll
            for (int off = TC / 2; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffff, sum, off);

            // paper's update, all in registers
            float m_new = fmaxf(m_reg[r], mx);
            float l_new = expf(m_reg[r] - m_new) * l_reg[r] + expf(mx - m_new) * sum;
            float inv_l = 1.f / l_new;
            a[r] = l_reg[r] * expf(m_reg[r] - m_new) * inv_l;
            float b = expf(mx - m_new) * inv_l;
            m_reg[r] = m_new;
            l_reg[r] = l_new;

            // publish P for the P V product
            #pragma unroll
            for (int c = 0; c < TN; c++)
                Sij[(tr * TN + r) * S_STRIDE + tc * TN + c] = s[r][c] * b;
        }
        // __syncwarp();
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < loads_KV; e++) {               // (7) V into the same buffer
            int idx = e * NT + tx;
            if (idx < Bc * d) {
                int row = idx / d, col = idx % d;
                if (j * Bc + row < N)
                    KVj[row * d + col] = V[qkv_off + (j * Bc + row) * d + col];
            }
        }
        __syncthreads();                                   // (8)

        #pragma unroll
        for (int r = 0; r < TN; r++)
            #pragma unroll
            for (int q = 0; q < DCH; q++)
                #pragma unroll
                for (int cc = 0; cc < TN; cc++)
                    o[q][r][cc] *= a[r];

        // P V: accumulate straight into the (already rescaled) o
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            float p[TN];
            #pragma unroll
            for (int r = 0; r < TN; r++) p[r] = Sij[(tr * TN + r) * S_STRIDE + c];
            #pragma unroll
            for (int q = 0; q < DCH; q++) {
                float v[TN];
                #pragma unroll
                for (int cc = 0; cc < TN; cc++) v[cc] = KVj[c * d + q * TC * TN + tc * TN + cc];
                #pragma unroll
                for (int r = 0; r < TN; r++)
                    #pragma unroll
                    for (int cc = 0; cc < TN; cc++)
                        o[q][r][cc] += p[r] * v[cc];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < TN; r++) {
        int grow = i * Br + tr * TN + r;
        if (grow < N)
            #pragma unroll
            for (int q = 0; q < DCH; q++)
                #pragma unroll
                for (int cc = 0; cc < TN; cc++)
                    O[qkv_off + grow * d + q * TC * TN + tc * TN + cc] = o[q][r][cc];
    }
}

template <int Br, int Bc, int TN, int d>
void setParamsAndCallKer(int N, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d, float*l_d) {
    constexpr int NT = (Br / TN) * (Bc / TN);
    dim3 block(NT);
    dim3 grid(CEIL_DIV(N, Br));

    const size_t shmem_size = SmemLayout<Br, Bc, d>::TOTAL * sizeof(float);

    gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, TN, d>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
    gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, TN, d>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    int blocks_per_sm;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, s2mm_kernel<Br, Bc, TN, d>, NT, shmem_size));
    fprintf(stderr, "d = %d: %d blocks per SM\n", d, blocks_per_sm);

    s2mm_kernel<Br, Bc, TN, d><<<grid, block, shmem_size>>>(N, Q_d, K_d, V_d, O_d, m_d, l_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{

    const int M = 164 * 1024;    
    const int Br = std::min(CEIL_DIV(M, 4 * d), d);


    if(d == 128) {
        setParamsAndCallKer<64,64,4,128>(N,Q_d,K_d,V_d,O_d,m_d, l_d);
    } else if (d == 64) {
        setParamsAndCallKer<64,64,4,64>(N,Q_d,K_d,V_d,O_d,m_d, l_d);
    } else {
        printf("Unsupported dataset, exiting!");
        exit(1);
    }

    // cudaDeviceSynchronize();

#if 0
    {
        float* m_h = (float*) malloc(N*sizeof(float));
        cudaMemcpy(m_h, m_d, N*sizeof(float), cudaMemcpyDeviceToHost);
        {
            printf("(N,d,BN,TN)=(%d,%d,%d,%d), ms:\n    ", N,d,BN,TN);
            for(int q=0; q<BN; q++) {
                printf(", %f", m_h[q]);
            }
            printf("\n");
        }

        free(m_h);
    }
#endif

    return 0;
}

int main(int argc, char **argv)
{
    int status = EXIT_SUCCESS;
    if (argc != 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s d N runs in_path out_path Compute with matrices filled with ones\n", argv[0]);        fprintf(stderr, "  %s d N runs in out Read Q, K, V from Futhark binary in, write O to out\n", argv[0]);
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

    struct timeval ts, te;
    double dur;
    {
        float *Q_d, *K_d, *V_d, *O_d, *m_d, *l_d;

        gpuAssert(cudaSetDevice(0));

        // allocate memory on device
        gpuAssert(cudaMalloc((void**) &Q_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &O_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &V_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &K_d, cnt*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &m_d, (size_t)N*sizeof(float)));
        gpuAssert(cudaMalloc((void**) &l_d, (size_t)N*sizeof(float)));

        // copy host memory to device
        gpuAssert(cudaMemcpy(Q_d, Q, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(V_d, V, cnt*sizeof(float), cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(K_d, K, cnt*sizeof(float), cudaMemcpyHostToDevice));

        fprintf(stderr, "Warming up...");

        
        s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);

        
        gpuAssert(cudaDeviceSynchronize());

        fprintf(stderr, "done.\n");
        fprintf(stderr, "Running s2mm...");

        fprintf(stderr, "Printing runtimes to stdout\n");

        for(int i=0; i<runs; i++) {
            gettimeofday(&ts, NULL);
            s2mm(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);
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

        /* QK^t is 2N^2d flops, so is PV. softmax(S) (row-wise)
         * exp(S[i]) / sum_j exp(P[i, j] - max(P[i]))
         * is N * (N + 4N) = 5 N^2 flops, but exp is more expensive. */
        fprintf(stderr,
                "Compute rate: %lf Gflops/s, runtime: %lf ms\n",
                4.0 * d * N * N / dur / 1e9,  dur*1e3);

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
