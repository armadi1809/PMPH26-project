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


float* allocate_host_memory(size_t cnt) {
    float * ptr;
    if ((ptr = (float*) calloc(cnt, sizeof(float))) == NULL) {
        return NULL;
    }

    return ptr;
}


template <int Br, int Bc, int Tile_r, int Tile_c, int mul>
__global__ void __launch_bounds__((Br * Bc) / (Tile_r * Tile_c), 1) // TODO: check if we need something else
s2mm_kernel(int N, int d, const float* Q, const float* K, const float* V, float* O, float* ms) {
    // Shared memory
    extern __shared__ float shared_memory[];
    float* shared_Q = shared_memory;
    float* shared_K = shared_memory + Br * Bc * mul;
    float* shared_P = shared_K + Br * (Bc+1);
    float* shared_V = shared_K; // TODO: decide if to reuse K memory for V

    float* maxs = shared_P + Br * Bc;   // [Br]
    float* sums = maxs + Br;            // [Br]
    float* es   = sums + Br;            // [Br]
    float* el   = es   + Br;            // [Br]
    float* li   = el   + Br;            // [Br]
    float* mi   = li   + Br;            // [Br]

    const int block_size = Br * Bc;
    const int tile_size = Tile_r * Tile_c;
    const int num_threads = block_size / tile_size;

    for (int t = threadIdx.x; t < Br; t+=num_threads) {
        mi[t] = -INFINITY;
        li[t] = 0;
    }

    float reg_O[mul][Tile_r][Tile_c] = {0.0};
    float reg_P[Tile_r][Tile_c] = {0.0};
    
    const float* ptr_Q;
    const float* ptr_K;
    const float* ptr_V;
    float* ptr_O;

    const int thread_row_inp = threadIdx.x / Br;
    const int thread_col_inp = threadIdx.x % Br;

    const int thread_row_out = threadIdx.x / (Br / Tile_r);
    const int thread_col_out = threadIdx.x % (Br / Tile_r);

    const int stride_inp = num_threads / Br;
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;

    ptr_Q = Q + block_row * Br * d;
    float *ptr_shared_Q = shared_Q;

    // copy Q into shared memory from HBM
    for (int i = threadIdx.x; i < Br * d; i += num_threads) { // TODO: maybe blockDim here?
        ptr_shared_Q[i] = ptr_Q[i];
    }

    ptr_K = K + block_col * Bc * d;
    float *ptr_shared_K = shared_K;

    // copy K into shared memory from HBM
    for (int i = threadIdx.x; i < Bc * d; i+= num_threads) {
        ptr_shared_K[i] = ptr_K[i];
    }

    __syncthreads();

    int cols = (Bc / Tile_c);

    int start_t_i = (threadIdx.x * Tile_c) / cols;
    int start_t_j = (threadIdx.x * Tile_c) % cols;

    // matrix multiply within tile
    for (int t_i = start_t_i; t_i < start_t_i + Tile_r; t_i++) {
        float t_max = 0.0;

        for (int t_j = start_t_j; t_j < start_t_j + Tile_c; t_j++) {
            // multiply vectors
            for (int k = 0; k < Br; k++) {
                reg_P[t_i][t_j] += ptr_shared_Q[t_i * d + k] * ptr_shared_K[t_j * d + k];
            }

            t_max = std::max(t_max, reg_P[t_i][t_j]);
        }

        float t_sum = 0.0;
        for (int t_j = start_t_j; t_j < start_t_j + Tile_c; t_j++) {
            t_sum += exp(reg_P[t_i][t_j] - t_max);
        }

        maxs[threadIdx.x / cols] = t_max;
        sums[threadIdx.x / cols] = t_sum;
    }

    // tile work should be done
    __syncthreads();
}


template<int Br, int Bc, int Tile_r, int Tile_c, int mul>
void setParamsAndCallKer(int N, int d, float *Q_d, float *K_d, float *V_d, float *O_d, float* m_d) {
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
    const int tile_size = Tile_r * Tile_c;

    dim3 grid(N / Br, N / Bc);
    dim3 block(block_size / tile_size);
    const size_t shmem_size = (
        Br * Bc * mul + // shared_Q
        Br * (Bc+1) +   // shared_K, shared_V
        Br * Bc +       // shared_P
        6*Br            // maxs, sums, es, el, li, mi
    ) * sizeof(float);

    // printf("N %d, d %d, BN %d, Bd %d, TN %d, mul %d, shmem %d\n", N, d, BN, Bd, TN, mul, shmem_size);

    gpuAssert(cudaFuncSetAttribute(s2mm_kernel<Br, Bc, Tile_r, Tile_c, mul>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

    s2mm_kernel<Br, Bc, Tile_r, Tile_c, mul><<<grid, block, shmem_size>>>(N, d, Q_d, K_d, V_d, O_d, m_d);
    gpuAssert(cudaPeekAtLastError());
}

__host__ int
s2mm(float* m_d, float* l_d, float *O_d, float *Q_d, float *K_d, float *V_d, int N, int d)
{

    const int M = 164 * 1024;

    if(d == 128) {
        const int Br = std::min((M + (4 * d) - 1) / d, d);
        const int Bc = (M + (4 * d) - 1) / d;
        const int Tile_r = 2;
        const int Tile_c = 2;

        setParamsAndCallKer<Br, Bc, Tile_r, Tile_c,2>(N,d,Q_d,K_d,V_d,O_d,m_d);
    } else if (d == 64) {
        const int Br = std::min((M + (4 * d) - 1) / d, d);
        const int Bc = (M + (4 * d) - 1) / d;
        const int Tile_r = 2;
        const int Tile_c = 2;

        setParamsAndCallKer<Br, Bc, Tile_r, Tile_c,1>(N,d,Q_d,K_d,V_d,O_d,m_d);
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
