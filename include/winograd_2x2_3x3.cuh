#ifndef WINOGRAD_2x2_3x3_CUH
#define WINOGRAD_2x2_3x3_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#include "utils.cuh"
#include "kernels/winograd_2x2_3x3_GgGT.cuh"
#include "kernels/winograd_2x2_3x3_BTdB.cuh"
#include "kernels/winograd_2x2_3x3_ATtA.cuh"

/*
    filter_transform = [4, 4, K, C]
    input_transform = [4, 4, C, N, th, tw]
    hadamard = [4, 4, K, N, th, tw]
    inverse transform = [4, 4, K, N, th, tw] -> [2, 2, th, tw] (per K, N) -> [N, K, H, W]
*/

float *convWinograd_2x2_3x3(const float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, int padding = 0)
{
    auto divUp = [](int x, int y)
    { return (x + y - 1) / y; };

    float *d_filter_transform;
    // computing  filter_transform

    float *d_F;
    {
        CUDA_CALL(cudaMalloc((void **)&d_F, K * C * 3 * 3 * sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_F, h_f, K * C * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void **)&d_filter_transform, 4 * 4 * K * C * sizeof(float)));

        const int NUM_KERNELS_PER_BLOCK = 40, BLOCK_SIZE = 128;
        dim3 grid(divUp(K * C, NUM_KERNELS_PER_BLOCK));

        winograd_2x2_3x3_GgGT<NUM_KERNELS_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_F, K, C, d_filter_transform);
    }

    float *d_inp_transform;
    // computing input_transform

    const int OUT_H = (H + 2 * padding - 3 + 1);
    const int OUT_W = (W + 2 * padding - 3 + 1);

    const int TILES_H = divUp(H + 2 * padding, 2);
    const int TILES_W = divUp(W + 2 * padding, 2);

    float *d_img;
    {
        CUDA_CALL(cudaMalloc((void **)&d_img, N * C * H * W * sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_img, h_img, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void **)&d_inp_transform, 4 * 4 * C * N * TILES_H * TILES_W * sizeof(float)));

        const int TILES_H_PER_BLOCK = 3,
                  TILES_W_PER_BLOCK = 16,
                  BLOCK_SIZE = 128;

        dim3 grid(divUp(TILES_W, TILES_W_PER_BLOCK),
                  divUp(TILES_H, TILES_H_PER_BLOCK),
                  N * C);

        winograd_2x2_3x3_BTdB<TILES_H_PER_BLOCK, TILES_W_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_img, N, C, H, W, d_inp_transform, TILES_H, TILES_W, padding);
    }

    // hadamard product
    float *d_M;
    {
        CUDA_CALL(cudaMalloc((void **)&d_M, 4 * 4 * K * N * TILES_H * TILES_W * sizeof(float)));

        cublasHandle_t cbls_handle;
        cublasCreate(&cbls_handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFree(d_F));
        CUDA_CALL(cudaFree(d_img));

        CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle,
                                              CUBLAS_OP_N, CUBLAS_OP_N,
                                              (N * TILES_H * TILES_W), K, C,
                                              &alpha,
                                              d_inp_transform, (N * TILES_H * TILES_W), (C * N * TILES_H * TILES_W),
                                              d_filter_transform, C, (K * C),
                                              &beta,
                                              d_M, (N * TILES_H * TILES_W), (K * N * TILES_H * TILES_W),
                                              16));
                                              
        CUDA_CALL(cudaFree(d_filter_transform));
        CUDA_CALL(cudaFree(d_inp_transform));
    }

    // inverse transfom
    float *d_out;

    {
        CUDA_CALL(cudaMalloc((void **)&d_out, N * K * OUT_H * OUT_W * sizeof(float)));

        const int NUM_TILES_PER_BLOCK = 45;
        const int BLOCK_SIZE = 128;

        dim3 grid(divUp(TILES_H * TILES_W, NUM_TILES_PER_BLOCK),
                  K,
                  N);
        winograd_2x2_3x3_ATtA<NUM_TILES_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_M, K, N, TILES_H, TILES_W, d_out, OUT_H, OUT_W);
    }

    float *h_out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_out, d_out, N * K * OUT_H * OUT_W * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_M));
    CUDA_CALL(cudaFree(d_out));

    return h_out;
}

#endif
