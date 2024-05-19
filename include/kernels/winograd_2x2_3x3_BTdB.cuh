#ifndef WINOGRAD_2x2_3x3_BTdB_CUH
#define WINOGRAD_2x2_3x3_BTdB_CUH

__device__ __forceinline__ void multiply_BT_2x2_3x3(const float in[4], float out[4])
{
    /*
        B = [
            1, 0, -1, 0
            0, 1, 1, 0
            0, -1, 1, 0
            0, 1, 0, -1
        ]
    */

    out[0] = in[0] - in[2];
    out[1] = in[1] + in[2];
    out[2] = in[2] - in[1];
    out[3] = in[1] - in[3];
}

template <int TILES_H_PER_BLOCK, int TILES_W_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_2x2_3x3_BTdB(
    const float *d, const int N, const int C, const int H, const int W,
    float *inp_transform, const int TILES_H, const int TILES_W,
    const int padding)
{
    /*
        Input: d = img [N, C, H, W]
        Output: input_transform [4, 4, C, N, TILES_H, TILES_W]
    */

    const int tile_col_idx_start = blockIdx.x * TILES_W_PER_BLOCK;
    const int tile_row_idx_start = blockIdx.y * TILES_H_PER_BLOCK;
    const int nc = blockIdx.z;
    const int c = nc % C;
    const int n = nc / C;

    // calculating elements needed to create [TILES_H_PER_BLOCK, TILES_W_PER_BLOCK] tiles.
    constexpr int INP_FRAME_H = TILES_H_PER_BLOCK * 2 + 2; // 3 - 1
    constexpr int INP_FRAME_W = TILES_W_PER_BLOCK * 2 + 2; // 3 - 1

    __shared__ float s_d[INP_FRAME_H][INP_FRAME_W];

    for (int i = threadIdx.x; i < INP_FRAME_H * INP_FRAME_W; i += BLOCK_SIZE)
    {
        const int s_col_idx = i % INP_FRAME_W;
        const int s_row_idx = i / INP_FRAME_W;

        const int col_idx = tile_col_idx_start * 2 - padding + s_col_idx;
        const int row_idx = tile_row_idx_start * 2 - padding + s_row_idx;

        if (row_idx >= 0 && row_idx < H && col_idx >= 0 && col_idx < W)
            s_d[s_row_idx][s_col_idx] = d[((n * C + c) * H + row_idx) * W + col_idx];
        else
            s_d[s_row_idx][s_col_idx] = 0;
    }
    __syncthreads();

    // computing BTd
    __shared__ float s_BTd[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][4][4];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        // getting a col
        const int col_idx = i % 4;
        const int tile_idx = i / 4;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_row_idx_start + tile_row_idx < TILES_H && tile_col_idx_start + tile_col_idx < TILES_W)
        {
            float in_col[4];
            for (int row_idx = 0; row_idx < 4; ++row_idx)
                in_col[row_idx] = s_d[tile_row_idx * 2 + row_idx][tile_col_idx * 2 + col_idx];

            float out_col[4];
            multiply_BT_2x2_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 4; ++row_idx)
                s_BTd[tile_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    __shared__ float s_BTdB[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][4 * 4];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        // getting a row
        const int row_idx = i % 4;
        const int tile_idx = i / 4;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_row_idx_start + tile_row_idx < TILES_H && tile_col_idx_start + tile_col_idx < TILES_W)
        {
            float in_row[4];
            for (int col_idx = 0; col_idx < 4; ++col_idx)
                in_row[col_idx] = s_BTd[tile_idx][row_idx][col_idx];

            float out_row[4];
            multiply_BT_2x2_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 4; ++col_idx)
                s_BTdB[tile_idx][row_idx * 4 + col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 4 * 4 * TILES_H_PER_BLOCK * TILES_W_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int tile_idx = i % (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);
        const int offset = i / (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        const int global_tile_col_idx = tile_col_idx_start + tile_col_idx;
        const int global_tile_row_idx = tile_row_idx_start + tile_row_idx;

        if (global_tile_row_idx < TILES_H && global_tile_col_idx < TILES_W)
            inp_transform[(((offset * C + c) * N + n) * TILES_H + global_tile_row_idx) * TILES_W + global_tile_col_idx] = s_BTdB[tile_idx][offset];
    }
    // __syncthreads();
}

#endif