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
    const int loads_KV = CEIL_DIV(Bc * d, NT);

    int tx = threadIdx.x;

    int Tc = CEIL_DIV(N, Bc);

    int qkv_off = 0;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * (d + 1);
    float* Sij = Vj + Bc * d;
    float* Oi = Sij + Br * S_STRIDE;

    float* li = Oi + Br * d;
    float* li_new = li + Br;

    float* mi = li_new + Br;
    float* mi_new = mi + Br;
    float* mij_dash = mi_new + Br;

    int i = blockIdx.x;

    // load Q_i
    const int elem_loads_per_thread_Q = CEIL_DIV(Br * d, NT);

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

    for (int t = threadIdx.x; t < Br * d; t += num_threads)
        Oi[t] = 0.f;

    for (int t = threadIdx.x; t < Br; t += num_threads) {
        mi[t] = -INFINITY;
        li[t] = 0.f;
    }

    __syncthreads();

    const int tr = tx / (Bc / TN);
    const int tc = tx % (Bc / TN);

    for (int j = 0; j < Tc; j++) {
        __syncthreads();
        
        for (int e = 0; e < loads_KV; e++) {
                int idx = e * NT + tx;                  // stride by NT, not Br * Bc
                if (idx < Bc * d) {
                    int row = idx / d;
                    int col = idx % d;
                    if (j * Bc + row < N) {
                        Kj[row * (d + 1) + col] = K[qkv_off + (j * Bc + row) * d + col];
                        Vj[row * d + col]       = V[qkv_off + (j * Bc + row) * d + col];
                    }
                }
            }
        __syncthreads();

        float s[TN][TN] = {0.f};
        for (int k = 0; k < d; k++) {
            float q[TN], kv[TN];
            #pragma unroll
            for (int r = 0; r < TN; r++) q[r]  = Qi[(tr * TN + r) * d + k];
            #pragma unroll
            for (int c = 0; c < TN; c++) kv[c] = Kj[(tc * TN + c) * (d + 1) + k];
            #pragma unroll
            for (int r = 0; r < TN; r++)
                #pragma unroll
                for (int c = 0; c < TN; c++)
                    s[r][c] += q[r] * kv[c];
        }

        #pragma unroll
        for (int r = 0; r < TN; r++)
            #pragma unroll
            for (int c = 0; c < TN; c++) {
                int key = j * Bc + tc * TN + c;              // global key index
                Sij[(tr * TN + r) * S_STRIDE + tc * TN + c] = (key < N) ? s[r][c] : -INFINITY;
            }
        
        __syncthreads();
        // softmax: one thread per row (only tx < Br do work, since Br = 64 < NT = 256)
        for (int row = tx; row < Br; row += NT) {
            float row_m = -INFINITY, row_l = 0.f;
            for (int c = 0; c < Bc; c++)
                row_m = fmaxf(row_m, Sij[row * S_STRIDE + c]);

            for (int c = 0; c < Bc; c++) {
                float exp_val = expf(Sij[row * S_STRIDE + c] - row_m);
                Sij[row * S_STRIDE + c] = exp_val;
                row_l += exp_val;
            }

            mij_dash[row] = row_m;
            mi_new[row]   = fmaxf(mi[row], row_m);
            li_new[row]   = expf(mi[row] - mi_new[row]) * li[row] +
                            expf(row_m - mi_new[row]) * row_l;
        }
        __syncthreads();

        // P V and O update: block-stride over all Br * d output elements
        for (int t = tx; t < Br * d; t += NT) {
            int row = t / d;
            int col = t % d;

            float acc = 0.f;
            for (int c = 0; c < Bc; c++)
                acc += Sij[row * S_STRIDE + c] * Vj[c * d + col];

            Oi[t] = (1.f / li_new[row]) *
                    (li[row] * expf(mi[row] - mi_new[row]) * Oi[t] +
                     expf(mij_dash[row] - mi_new[row]) * acc);
        }
        __syncthreads();

        for (int row = tx; row < Br; row += NT) {
            mi[row] = mi_new[row];
            li[row] = li_new[row];
        }
    }

    for (int t = tx; t < Br * d; t += num_threads) {
        int row = t / d;
        if (i * Br + row < N)
            O[qkv_off + (i * Br) * d + t] = Oi[t];
    }
}

template <int Br, int Bc, int TN, int d>
void setParamsAndCallKer(int N, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d, float*l_d) {
    // Assumptions: ( TN | BN ) && ( BN | d ) && ( TN^2 | BN ) && 
    //              ( (BN / TN^2) | BN ) && ( BN >= TN^2 ) && (BN | N)
    // TODO: come to this back later and fix it
    // bool assert1 = (BN % TN) == 0;
    // bool assert2 = (d % BN)  == 0;
    // bool assert3 = (BN % (TN*TN)) == 0;
    // bool assert4 = (BN % (BN / (TN*TN)))  == 0;
    // bool assert5 = (BN >= (TN*TN));
    // assert(assert1 && assert2 && assert3 && assert4 && assert5);
    constexpr int NT = (Br / TN) * (Bc / TN);
    dim3 block(NT);
    dim3 grid(CEIL_DIV(N, Br));     // CEIL_DIV, consistent with the kernel's bounds checks

    const size_t shmem_size = (
        Br * d +            // Qi
        Bc * (d + 1) +      // Kj   (padded rows)
        Bc * d +            // Vj
        Br * (Bc + 1) +     // Sij  (padded rows)
        Br * d +            // Oi
        5 * Br              // li, li_new, mi, mi_new, mij_dash
    ) * sizeof(float);
    // printf("N %d, d %d, BN %d, Bd %d, TN %d, mul %d, shmem %d\n", N, d, BN, Bd, TN, mul, shmem_size);

    gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, TN, d>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

    s2mm_kernel<Br, Bc, TN, d><<<grid, block, shmem_size>>>(N, Q_d, K_d, V_d, O_d, m_d, l_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{

    const int M = 164 * 1024;    
    const int Br = std::min(CEIL_DIV(M, 4 * d), d);
    const int Bc = CEIL_DIV(M, 4 *d);
    // Tr blocks!


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
