#ifndef WINOGRAD_2x2_3x3_GgGT_CUH
#define WINOGRAD_2x2_3x3_GgGT_CUH

__device__ __forceinline__ void multiply_G_2x2_3x3(const float in[3], float out[4])
{
    /*
        G = [
                1,    0,   0
                1/2,  1/2, 1/2
                1/2, -1/2, 1/2
                0,    0,   1
            ]
    */

    out[0] = in[0];
    // out[1] = 0.5 * (in[0] + in[1] + in[2]);
    // out[2] = 0.5 * (in[0] - in[1] + in[2]);
    float tmp = in[0] + in[2];
    out[1] = 0.5 * (tmp + in[1]);
    out[2] = 0.5 * (tmp - in[1]);
    out[3] = in[2];
}

template <int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_2x2_3x3_GgGT(
    const float *g, const int K, const int C,
    float *filter_transform)
{
    /*
        input-> g (filter) [K, C, 3, 3]
        output-> filter_transform [4, 4, K, C]
    */

    const int NUM_KERNELS = K * C; // total [3, 3] kernels

    const int kc_idx_start = blockIdx.x * NUM_KERNELS_PER_BLOCK;

    __shared__ float s_g[NUM_KERNELS_PER_BLOCK][4][4]; // [4, 4] - we will store output as well in it.

    // loading in shared memory
    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3 * 3; i += BLOCK_SIZE)
    {
        const int col_idx = i % 3;
        const int row_idx = (i / 3) % 3;
        const int kc_idx = (i / 3) / 3;

        const int global_kc_idx = kc_idx + kc_idx_start;

        if (global_kc_idx < NUM_KERNELS)
            s_g[kc_idx][row_idx][col_idx] = g[global_kc_idx * 9 + row_idx * 3 + col_idx];
    }

    __syncthreads();

    // computing Gg
    __shared__ float s_Gg[NUM_KERNELS_PER_BLOCK][4][3];

    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3; i += BLOCK_SIZE)
    {
        // will fetch a col here, 1x3 and multiply above
        const int col_idx = i % 3;
        const int kc_idx = i / 3;

        if (kc_idx_start + kc_idx < NUM_KERNELS)
        {
            float in_col[3];

            for (int row_idx = 0; row_idx < 3; ++row_idx)
                in_col[row_idx] = s_g[kc_idx][row_idx][col_idx];

            float out_col[4];
            multiply_G_2x2_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 4; ++row_idx)
                s_Gg[kc_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    // computing GgGT
    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        // will fetch a row from Gg
        const int row_idx = i % 4;
        const int kc_idx = i / 4;

        if (kc_idx_start + kc_idx < NUM_KERNELS)
        {
            float in_row[3];

            for (int col_idx = 0; col_idx < 3; ++col_idx)
                in_row[col_idx] = s_Gg[kc_idx][row_idx][col_idx];

            float out_row[4];
            multiply_G_2x2_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 4; ++col_idx)
                s_g[kc_idx][row_idx][col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    // loading results to GMem

    for (int i = threadIdx.x; i < 4 * 4 * NUM_KERNELS_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int kc_idx = i % NUM_KERNELS_PER_BLOCK;
        const int offset = (i / NUM_KERNELS_PER_BLOCK);

        const int col_idx = offset % 4;
        const int row_idx = offset / 4;

        const int global_kc_idx = kc_idx_start + kc_idx;
        if (global_kc_idx < NUM_KERNELS)
            filter_transform[offset * NUM_KERNELS + global_kc_idx] = s_g[kc_idx][row_idx][col_idx];
    }
}

#endif