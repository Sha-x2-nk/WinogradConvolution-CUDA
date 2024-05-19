#ifndef WINOGRAD_4x4_3x3_GgGT_CUH
#define WINOGRAD_4x4_3x3_GgGT_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ void multiply_G_4x4_3x3(const float in[3], float out[6])
{
    // G = [
    //      1/4.0,      0,       0,
    //     -1/6.0,  -1/6.0, -1/6.0,
    //     -1/6.0,   1/6.0, -1/6.0,
    //     1/24.0,  1/12.0,  1/6.0,
    //     1/24.0, -1/12.0,  1/6.0,
    //          0,       0,      1
    // ]

    out[0] = in[0] / 4;
    float tmp0 = (-in[0] - in[2]);
    out[1] = (tmp0 - in[1]) / 6;
    out[2] = (tmp0 + in[1]) / 6;

    tmp0 = in[0] / 24;
    float tmp1 = in[1] / 12;
    float tmp2 = in[2] / 6;

    out[3] = tmp0 + tmp1 + tmp2;
    out[4] = tmp0 - tmp1 + tmp2;
    out[5] = in[2];
}

template <int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_4x4_3x3_GgGT(
    const float *g, const int K, const int C,
    float *filter_transform)
{
    /*
        input -> g (filter): [K, C, 3, 3]
        output -> filter_transform : [6, 6, K, C]
    */

    const int NUM_KERNELS = K * C;

    const int kc_idx_start = blockIdx.x * NUM_KERNELS_PER_BLOCK;

    // will store our input first.
    // 6x6 in order to reuse this to store outputs later
    __shared__ float shared_6x6[NUM_KERNELS_PER_BLOCK][6][6];

    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3 * 3; i += BLOCK_SIZE)
    {
        const int col_idx = i % 3;
        const int row_idx = (i / 3) % 3;
        const int kc_idx = (i / 9);

        const int global_kc = kc_idx + kc_idx_start;
        if (global_kc < NUM_KERNELS)
            shared_6x6[kc_idx][row_idx][col_idx] = g[global_kc * 9 + row_idx * 3 + col_idx];
    }

    __syncthreads();

    // Gg
    __shared__ float shared_6x3[NUM_KERNELS_PER_BLOCK][6][3];

    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3; i += BLOCK_SIZE)
    {
        // will fetch a col here, 1x3 and multiply above
        const int col_idx = i % 3;
        const int kc_idx = i / 3;

        if (kc_idx_start + kc_idx < NUM_KERNELS)
        {
            float in_col[3];

            for (int row_idx = 0; row_idx < 3; ++row_idx)
                in_col[row_idx] = shared_6x6[kc_idx][row_idx][col_idx];

            float out_col[6];
            multiply_G_4x4_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 6; ++row_idx)
                shared_6x3[kc_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    // GgGT
    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // will fetch a row from Gg
        const int row_idx = i % 6;
        const int kc_idx = i / 6;

        if (kc_idx + kc_idx_start < NUM_KERNELS)
        {
            float in_row[3];
            for (int col_idx = 0; col_idx < 3; ++col_idx)
                in_row[col_idx] = shared_6x3[kc_idx][row_idx][col_idx];

            float out_row[6];
            multiply_G_4x4_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 6; ++col_idx)
                shared_6x6[kc_idx][row_idx][col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    // loading back to GMem
    for (int i = threadIdx.x; i < 6 * 6 * NUM_KERNELS_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int kc_idx = i % NUM_KERNELS_PER_BLOCK;
        const int offset = i / NUM_KERNELS_PER_BLOCK;

        const int col_idx = offset % 6;
        const int row_idx = offset / 6;

        const int global_kc = kc_idx_start + kc_idx;
        if (global_kc < NUM_KERNELS)
            filter_transform[offset * NUM_KERNELS + global_kc] = shared_6x6[kc_idx][row_idx][col_idx];
    }
}

#endif