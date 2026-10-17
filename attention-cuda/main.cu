#include <assert.h>
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

        

        TODO(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);

        
        gpuAssert(cudaDeviceSynchronize());

        fprintf(stderr, "done.\n");
        fprintf(stderr, "Running s2mm...");

        fprintf(stderr, "Printing runtimes to stdout\n");

        for(int i=0; i<runs; i++) {
            gettimeofday(&ts, NULL);
            TODO(m_d, l_d, O_d, Q_d, K_d, V_d, N, d);
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
