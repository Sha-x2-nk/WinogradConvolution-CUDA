## Phases
Non fused winograd algorithm works in 4 phases.
1. <b>Filter Transform</b>: Transform the filters from spatial domain to Winograd domain. [GgGT]
```cpp
// computing filter transform
float *d_filter_transform;
float *d_F;
CUDA_CALL(cudaMalloc((void **)&d_F, K * C * 3 * 3 * sizeof(float)));
CUDA_CALL(cudaMemcpy(d_F, h_f, K * C * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

CUDA_CALL(cudaMalloc((void **)&d_filter_transform, 6 * 6 * K * C * sizeof(float)));

const int NUM_KERNELS_PER_BLOCK = 10, 
          BLOCK_SIZE = 128;

dim3 grid(divUp(K * C, 
          NUM_KERNELS_PER_BLOCK)
        );

winograd_4x4_3x3_GgGT<NUM_KERNELS_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_F, K, C, d_filter_transform);
```
2. <b>Input Transform</b>: Transform the input from spatial domain to Winograd domain.[BTdB]
```cpp
float *d_inp_transform;
float *d_img;
CUDA_CALL(cudaMalloc((void **)&d_inp_transform, 6 * 6 * C * N * TILES_Y * TILES_X * sizeof(float)));

CUDA_CALL(cudaMalloc((void **)&d_img, N * C * H * W * sizeof(float)));
CUDA_CALL(cudaMemcpy(d_img, h_img, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

const int TILES_Y_PER_BLOCK = 4, 
          TILES_X_PER_BLOCK = 8, 
          BLOCK_SIZE = 128;

dim3 grid(divUp(TILES_X, TILES_X_PER_BLOCK),
          divUp(TILES_Y, TILES_Y_PER_BLOCK),
          N * C
        );

winograd_4x4_3x3_BTdB<TILES_Y_PER_BLOCK, TILES_X_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_img, N, C, H, W, d_inp_transform, TILES_Y, TILES_X, padding);
```
3. <b>Batched GEMM for Hadamard Product</b>: Perform batched General Matrix Multiply (GEMM) for the Hadamard product of the transformed input and filters. For this, we did not write custom kernel, but used cuBlas.
```cpp
// hadamard
float *d_M;
CUDA_CALL(cudaMalloc((void **)&d_M, 6 * 6 * K * N * TILES_Y * TILES_X * sizeof(float)));
cublasHandle_t cbls_handle;
cublasCreate(&cbls_handle);

const float alpha = 1.0f,
            beta = 0.0f;

CUDA_CALL(cudaDeviceSynchronize()); // make sure filter and input transforms are ready
CUDA_CALL(cudaFree(d_img));
CUDA_CALL(cudaFree(d_F));
CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        (N * TILES_Y * TILES_X), K, C,
                                        &alpha,
                                        d_inp_transform, (N * TILES_Y * TILES_X), (C * N * TILES_Y * TILES_X),
                                        d_filter_transform, C, (K * C),
                                        &beta,
                                        d_M, (N * TILES_Y * TILES_X), (K * N * TILES_Y * TILES_X),
                                        6 * 6));

CUDA_CALL(cudaFree(d_filter_transform));
CUDA_CALL(cudaFree(d_inp_transform));
```
4. <b>Inverse Transform</b>: Transform the output back from Winograd domain to spatial domain.[ATtA]
```cpp
// inverse transform
float *d_out;
const int OUT_H = (H + 2 * padding - 3 + 1),
          OUT_W = (W + 2 * padding - 3 + 1);
CUDA_CALL(cudaMalloc((void **)&d_out, N * K * OUT_H * OUT_W * sizeof(float)));
const int NUM_TILES_PER_BLOCK = 32,
          BLOCK_SIZE = 128;

dim3 grid(divUp(TILES_Y * TILES_X, NUM_TILES_PER_BLOCK), 
          K, 
          N
        );

winograd_4x4_3x3_ATtA<NUM_TILES_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_M, N, K, TILES_Y, TILES_X, d_out, OUT_H, OUT_W);
```
