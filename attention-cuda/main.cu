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


template <int Br, int Bc, int mul>
// __global__ void __launch_bounds__((Br * Bc) / (Tile_r * Tile_c), 1) // TODO: check if we need something else
__global__ void __launch_bounds__(Br * Bc) s2mm_kernel(int N, int d, const float* Q, const float* K, const float* V, float* O, float *m, float*l) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);

    // Single head: blocks split the row tiles i, all read the same Q, K, V.
    int qkv_off = 0;
    int lm_off = 0;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * (d + 1);
    float* Sij = Vj + Bc * d;
    float* Oi = Sij + Br * Bc;
    float* li = Oi + Br * d;
    float* li_new = li + Br;
    float* mi = li_new + Br;
    float* mi_new = mi + Br;
    float* mij_dash = mi_new + Br;

    for (int j = 0; j < Tc; j++) {
        __syncthreads();
        int loads_per_thread = CEIL_DIV(d, Br);
        for (int e = 0; e < loads_per_thread; e++) {
            int idx = e * (Br * Bc) + tx;
            if (idx < Bc * d) {
                int row = idx / d;
                int col = idx % d;

                if (j * Bc + row < N) {
                    Kj[row * (d + 1) + col] = K[qkv_off + (j * Bc + row) * d + col];
                    Vj[row * d + col] = V[qkv_off + (j * Bc + row) * d + col];
                }
            }
        }
        __syncthreads();

        for (int i = bx; i < Tr; i += gridDim.x) {
            for (int e = 0; e < loads_per_thread; e++) {
                int idx = e * (Br * Bc) + tx;
                if (idx < Br * d) {
                    int row = idx / d;
                    int col = idx % d;
                    if (i * Br + row < N) {
                        Qi[row * d + col] = Q[qkv_off + (i * Br + row) * d + col];
                        Oi[row * d + col] = j == 0 ? 0.f : O[qkv_off + (i * Br + row) * d + col];
                    }
                }
            }
        
            int s_row = tx / Bc;
            int s_col = tx % Bc;

            if (s_col == 0) {
                mi[s_row] = j == 0 ? -INFINITY : m[lm_off + (i * Br) + s_row];
                li[s_row] = j == 0 ? 0.f : l[lm_off + (i * Br) + s_row];
            }
            __syncthreads();

            float acc = 0.f;
            for (int k = 0; k < d; k++)
                acc += Qi[s_row * d + k] * Kj[s_col * (d + 1) + k];

            Sij[s_row * Bc + s_col] = acc;
            __syncthreads();

            if (s_col == 0) {
                float row_m = -INFINITY, row_l = 0.f;
                // Find max for numerical stability
                for (int c = 0; c < Bc; c++) {
                    float val = Sij[s_row * Bc + c];
                    if (val > row_m) {
                        row_m = val;
                    }
                }
                // Compute exponentials and sum
                for (int c = 0; c < Bc; c++) {
                    float exp_val = expf(Sij[s_row * Bc + c] - row_m);
                    Sij[s_row * Bc + c] = exp_val;
                    row_l += exp_val;
                }

                // Update running max and sum
                mij_dash[s_row] = row_m;
                mi_new[s_row] = max(mi[s_row], row_m);
                li_new[s_row] = expf(mi[s_row] - mi_new[s_row]) * li[s_row] + 
                                expf(row_m - mi_new[s_row]) * row_l;
            }
            __syncthreads();

            for (int col = s_col; col < d; col += Bc) {
                float acc = 0.f;
                for (int c = 0; c < Bc; c++)
                    acc += Sij[s_row * Bc + c] * Vj[c * d + col];

                int global_row = (i * Br) + s_row;
                if (global_row < N) {
                    Oi[s_row * d + col] = (1 / li_new[s_row]) * 
                        ((li[s_row] * expf(mi[s_row] - mi_new[s_row]) * Oi[s_row * d + col]) + 
                         (expf(mij_dash[s_row] - mi_new[s_row]) * acc));
                    O[qkv_off + global_row * d + col] = Oi[s_row * d + col];
                }
            }

            m[lm_off + (i * Br) + s_row] = mi_new[s_row];
            l[lm_off + (i * Br) + s_row] = li_new[s_row];
            __syncthreads();
        }
    }
}

template<int Br, int Bc, int mul>
void setParamsAndCallKer(int N, int d, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d, float*l_d) {
    // Assumptions: ( TN | BN ) && ( BN | d ) && ( TN^2 | BN ) && 
    //              ( (BN / TN^2) | BN ) && ( BN >= TN^2 ) && (BN | N)
    // TODO: come to this back later and fix it
    // bool assert1 = (BN % TN) == 0;
    // bool assert2 = (d % BN)  == 0;
    // bool assert3 = (BN % (TN*TN)) == 0;
    // bool assert4 = (BN % (BN / (TN*TN)))  == 0;
    // bool assert5 = (BN >= (TN*TN));
    // assert(assert1 && assert2 && assert3 && assert4 && assert5);

    const int block_size = Br * Bc;

    dim3 grid(N / Br);
    dim3 block(Br * Bc);
    const size_t shmem_size = (
        Br * d +        // Qi
        d * Br + (d + 1) * Bc +    // Kj, Vj
        Br * Bc +       // Sij
        Br * d +        // Oi
        5 * Br          // li, li_new, mi, mi_new, mij_dash
    ) * sizeof(float);

    // printf("N %d, d %d, BN %d, Bd %d, TN %d, mul %d, shmem %d\n", N, d, BN, Bd, TN, mul, shmem_size);

    gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, mul>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

    s2mm_kernel<Br, Bc, mul><<<grid, block, shmem_size>>>(N, d, Q_d, K_d, V_d, O_d, m_d, l_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{

    const int M = 164 * 1024;    
    const int Br = std::min(CEIL_DIV(M, 4 * d), d);
    const int Bc = CEIL_DIV(M, 4 *d);


    if(d == 128) {
        setParamsAndCallKer<32,32,2>(N,d,Q_d,K_d,V_d,O_d,m_d, l_d);
    } else if (d == 64) {
        setParamsAndCallKer<32,32,1>(N,d,Q_d,K_d,V_d,O_d,m_d, l_d);
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
