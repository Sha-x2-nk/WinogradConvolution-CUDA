#ifndef WINOGRAD_4x4_3x3_BTdB_CUH
#define WINOGRAD_4x4_3x3_BTdB_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ void multiply_BT_4x4_3x3(const float in[6], float out[6])
{
    /*
        BT = [
                4,  0, -5,  0, 1, 0,
                0, -4, -4,  1, 1, 0,
                0,  4, -4, -1, 1, 0,
                0, -2, -1,  2, 1, 0,
                0,  2, -1, -2, 1, 0,
                0,  4,  0, -5, 0, 1
             ]
    */

    int tmp0 = in[3] - 4 * in[1];
    int tmp1 = in[4] - 4 * in[2];

    out[0] = 4 * in[0] - 5 * in[2] + in[4];
    out[1] = tmp0 + tmp1;
    out[2] = tmp1 - tmp0;

    tmp0 = 2 * (in[1] - in[3]);
    tmp1 = in[4] - in[2];

    out[3] = tmp1 - tmp0;
    out[4] = tmp0 + tmp1;
    out[5] = 4 * in[1] - 5 * in[3] + in[5];
}

template <int TILES_H_PER_BLOCK, int TILES_W_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_4x4_3x3_BTdB(
    const float *d, const int N, const int C, const int H, const int W,
    float *inp_transform, const int TILES_H, const int TILES_W,
    const int padding)
{
    /*
        Input: d [N, C, H, W]
        Ouput: inp_transform [6, 6, C, N, TILES_H, TILES_W]
    */
    const int tile_col_idx_start = blockIdx.x * TILES_W_PER_BLOCK;
    const int tile_row_idx_start = blockIdx.y * TILES_H_PER_BLOCK;
    const int nc = blockIdx.z; // channels
    const int c = nc % C;
    const int n = nc / C;

    constexpr int INPUT_FRAME_H = TILES_H_PER_BLOCK * 4 + 2;
    constexpr int INPUT_FRAME_W = TILES_W_PER_BLOCK * 4 + 2;

    __shared__ float s_d[INPUT_FRAME_H][INPUT_FRAME_W];

    for (int i = threadIdx.x; i < INPUT_FRAME_H * INPUT_FRAME_W; i += BLOCK_SIZE)
    {
        const int s_col_idx = i % INPUT_FRAME_W;
        const int s_row_idx = i / INPUT_FRAME_W;

        const int row_idx = tile_row_idx_start * 4 - padding + s_row_idx;
        const int col_idx = tile_col_idx_start * 4 - padding + s_col_idx;

        if (row_idx >= 0 && row_idx < H && col_idx >= 0 && col_idx < W)
            s_d[s_row_idx][s_col_idx] = d[((n * C + c) * H + row_idx) * W + col_idx];
        else
            s_d[s_row_idx][s_col_idx] = 0;
    }
    __syncthreads();

    // computing BTd
    __shared__ float BTd[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][6][6];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // getting a col
        const int col_idx = i % 6;
        const int tile_idx = i / 6;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_row_idx_start + tile_row_idx < TILES_H && tile_col_idx_start + tile_col_idx < TILES_W)
        {
            float in_col[6];
            for (int row_idx = 0; row_idx < 6; ++row_idx)
                in_col[row_idx] = s_d[tile_row_idx * 4 + row_idx][tile_col_idx * 4 + col_idx];

            float out_col[6];
            multiply_BT_4x4_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 6; ++row_idx)
                BTd[tile_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    __shared__ float BTdB[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][36];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // getting a row
        const int row_idx = i % 6;
        const int tile_idx = i / 6;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_col_idx_start + tile_col_idx < TILES_W && tile_row_idx_start + tile_row_idx < TILES_H)
        {
            float in_row[6];
            for (int col_idx = 0; col_idx < 6; ++col_idx)
                in_row[col_idx] = BTd[tile_idx][row_idx][col_idx];

            float out_row[6];
            multiply_BT_4x4_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 6; ++col_idx)
                BTdB[tile_idx][row_idx * 6 + col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 6 * 6 * TILES_H_PER_BLOCK * TILES_W_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int tile_idx = i % (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);
        const int offset = i / (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        const int global_tile_col_idx = tile_col_idx + tile_col_idx_start;
        const int global_tile_row_idx = tile_row_idx + tile_row_idx_start;
        if (global_tile_col_idx < TILES_W && global_tile_row_idx < TILES_H)
            inp_transform[(((offset * C + c) * N + n) * TILES_H + global_tile_row_idx) * TILES_W + global_tile_col_idx] = BTdB[tile_idx][offset];
    }
}

#endif