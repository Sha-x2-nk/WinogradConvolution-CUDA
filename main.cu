#include <stdio.h>
#include <random>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "winograd_4x4_3x3.cuh"
#include "winograd_2x2_3x3.cuh"

const int N = 20,
          C = 512,
          H = 28,
          W = 28,
          K = 512,
          padding = 0,
          FILTER_SIZE = 3,
          FILTER_RADIUS = FILTER_SIZE / 2,
          OUT_H = H + 2 * padding - FILTER_SIZE + 1,
          OUT_W = W + 2 * padding - FILTER_SIZE + 1;

void initArr(float *A, int W);
float *convCPU(float *img, float *F);
float maxErr(float *A, float *B, int N);
void printArr(float *A, int N, int C, int H, int W);
float *benchmark_cudnn(const float *h_img, const float *h_f, cudnnConvolutionFwdAlgo_t algo);
int main()
{
    float *img = (float *)malloc(N * C * H * W * sizeof(float));
    initArr(img, N * C * H * W);
    // printf("\n IMG: ");
    // printArr(img, N, C, H, W);

    float *f = (float *)malloc(K * C * FILTER_SIZE * FILTER_SIZE * sizeof(float));
    initArr(f, K * C * FILTER_SIZE * FILTER_SIZE);
    // printf("\n FILTER: ");
    // printArr(f, K, C, FILTER_SIZE, FILTER_SIZE);

    float *out_actual = convCPU(img, f);
    // printf("\n ACTUAL OUT: ");
    // printArr(out_actual, N, K, OUT_H, OUT_W);

    // printf("\n CALC OUT: ");
    float *out = convWinograd_2x2_3x3(img, N, C, H, W, f, K, padding);
    // printArr(out, N, K, OUT_H, OUT_W);
    printf("\nMAX ERR WINOGRAD 2x2 3x3 : %f", maxErr(out, out_actual, N * K * OUT_H * OUT_W));
    free(out);

    out = convWinograd_4x4_3x3(img, N, C, H, W, f, K, padding);
    // printArr(out, N, K, OUT_H, OUT_W);
    printf("\nMAX ERR WINOGRAD 4x4 3x3 : %f", maxErr(out, out_actual, N * K * OUT_H * OUT_W));
    free(out);

    out = benchmark_cudnn(img, f, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
    // printArr(out, N, K, OUT_H, OUT_W);
    printf("\nMAX ERR CUDNN WINOGRADunfused : %f", maxErr(out, out_actual, N * K * OUT_H * OUT_W));
    free(out);

    free(img);
    free(f);
    free(out_actual);

    cudaDeviceReset();
    return 0;
}

void initArr(float *A, int W)
{
    for (int i = 0; i < W; ++i)
        A[i] = rand() % 10 + 1;
}

// only works for padding 0, 1
float *convCPU(float *img, float *F)
{
    float *out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            for (int h = FILTER_RADIUS - padding; h < H - (FILTER_RADIUS - padding); ++h)
                for (int w = FILTER_RADIUS - padding; w < W - (FILTER_RADIUS - padding); ++w)
                {
                    float tmp = 0;
                    for (int c = 0; c < C; ++c)
                        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i)
                            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j)
                                if (h + i >= 0 && h + i < H && w + j >= 0 && w + j < W)
                                    tmp += img[((n * C + c) * H + h + i) * W + (w + j)] * F[((k * C + c) * FILTER_SIZE + (i + FILTER_RADIUS)) * FILTER_SIZE + (j + FILTER_RADIUS)];
                    out[((n * K + k) * OUT_H + h - (FILTER_RADIUS - padding)) * OUT_W + w - (FILTER_RADIUS - padding)] = tmp;
                }
    return out;
}

float maxErr(float *A, float *B, int W)
{
    float maxErr = INT_MIN;
    for (int i = 0; i < W; ++i)
        maxErr = std::max<float>(abs(B[i] - A[i]), maxErr);

    return maxErr;
}

void printArr(float *A, int N, int C, int H, int W)
{
    printf("\n");

    for (int n = 0; n < N; ++n)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                for (int c = 0; c < C; ++c)
                    printf("%f,", A[(n * C + c) * H * W + h * W + w]);
                printf(" || ");
            }
            printf("\n");
        }
        printf("\n===================================================\n");
    }

    printf("\n");
}

float *benchmark_cudnn(const float *h_img, const float *h_f, cudnnConvolutionFwdAlgo_t algo)
{
    float *d_img;
    CUDA_CALL(cudaMalloc((void **)&d_img, N * C * H * W * sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_img, h_img, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

    float *d_f;
    CUDA_CALL(cudaMalloc((void **)&d_f, K * C * 3 * 3 * sizeof(float)));
    CUDA_CALL(cudaMemcpy(d_f, h_f, K * C * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

    float *d_out;
    CUDA_CALL(cudaMalloc((void **)&d_out, N * K * OUT_H * OUT_W * sizeof(float)));

    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    cudnnTensorDescriptor_t inputDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, 3, 3));

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(convDesc, 3));

    cudnnTensorDescriptor_t outputDesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, OUT_H, OUT_W));

    size_t workspaceSize = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));

    float *workspacePtr = nullptr;
    CUDA_CALL(cudaMalloc(&workspacePtr, workspaceSize));

    float alpha = 1.0, beta = 0.0;
    CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, inputDesc, d_img, filterDesc, d_f, convDesc, algo, workspacePtr, workspaceSize, &beta, outputDesc, d_out));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(workspacePtr));
    CUDA_CALL(cudaFree(d_img));
    CUDA_CALL(cudaFree(d_f));

    float *h_out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_out, d_out, N * K * OUT_H * OUT_W * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_out));

    return h_out;
}
