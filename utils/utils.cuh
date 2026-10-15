#pragma once

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

static __host__ void gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }
}

static bool parse_positive_int(const char *text, int *value) {
    char *end;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;
    *value = (int)parsed;
    return true;
}

static bool read_futhark_f32(FILE *f, float *dst, size_t cnt, int rank) {
    char head[7];
    int64_t dims[3];
    size_t total = 1;
    if (fread(head, 1, 7, f) != 7 || memcmp(head, "b\x02", 2) != 0 ||
        head[2] != rank || memcmp(head + 3, " f32", 4) != 0 ||
        fread(dims, sizeof(int64_t), rank, f) != (size_t)rank)
        return false;
    for (int i = 0; i < rank; i++)
        total *= (size_t)dims[i];
    return total == cnt && fread(dst, sizeof(float), cnt, f) == cnt;
}

static bool read_inputs(const char *path, float *Q, float *K, float *V, size_t cnt) {
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        return false;
    bool ok = read_futhark_f32(f, Q, cnt, 3) &&
              read_futhark_f32(f, K, cnt, 2) &&
              read_futhark_f32(f, V, cnt, 2);
    fclose(f);
    return ok;
}

static bool write_output(const char *path, const float *O, int N, int d) {
    FILE *f = fopen(path, "wb");
    if (f == NULL)
        return false;
    int64_t dims[3] = {N / d, d, d};
    size_t cnt = (size_t)N * (size_t)d;
    bool ok = fwrite("b\x02\x03 f32", 1, 7, f) == 7 &&
              fwrite(dims, sizeof(int64_t), 3, f) == 3 &&
              fwrite(O, sizeof(float), cnt, f) == cnt;
    return fclose(f) == 0 && ok;
}
