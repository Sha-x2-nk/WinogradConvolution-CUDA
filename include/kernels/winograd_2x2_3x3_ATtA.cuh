#ifndef WINOGRAD_2x2_3x3_ATtA_CUH
#define WINOGRAD_2x2_3x3_ATtA_CUH

__device__ __forceinline__ void multiply_AT_2x2_3x3(const float in[4], float out[2])
{
    out[0] = in[0] + in[1] + in[2];
    out[1] = in[1] - (in[2] + in[3]);
}

template <int NUM_TILES_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_2x2_3x3_ATtA(
    const float *t, const int K, const int N, const int TILES_H, const int TILES_W,
    float *out, const int OUT_H, const int OUT_W)
{
    /*
        Input: hadamard, [4 x 4 x K x N x TILES_Y x TILES_X]
        Output: conv output, [N x K x MAP_H x MAP_W]
    */
    const int NUM_TILES = TILES_H * TILES_W;

    const int tile_idx_start = blockIdx.x * NUM_TILES_PER_BLOCK;
    const int k = blockIdx.y;
    const int n = blockIdx.z;

    __shared__ float shared_4x4[NUM_TILES_PER_BLOCK][4 * 4];
    for (int i = threadIdx.x; i < 4 * 4 * NUM_TILES_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int local_tile_idx = i % NUM_TILES_PER_BLOCK;
        const int global_tile_idx = local_tile_idx + tile_idx_start;

        const int offset = i / NUM_TILES_PER_BLOCK;

        if (global_tile_idx < NUM_TILES)
            shared_4x4[local_tile_idx][offset] = t[((offset * K + k) * N + n) * NUM_TILES + global_tile_idx];
    }
    __syncthreads();

    // computing ATt
    __shared__ float shared_2x4[NUM_TILES_PER_BLOCK][2][4];
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        // take out a col
        const int col_idx = i % 4;
        const int local_tile_idx = i / 4;

        if (tile_idx_start + local_tile_idx < NUM_TILES)
        {
            float in_col[4];
            for (int row_idx = 0; row_idx < 4; ++row_idx)
                in_col[row_idx] = shared_4x4[local_tile_idx][row_idx * 4 + col_idx];

            float out_col[2];
            multiply_AT_2x2_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 2; ++row_idx)
                shared_2x4[local_tile_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    // will now compute ATtA
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 2; i += BLOCK_SIZE)
    {
        // take out a row
        const int row_idx = i % 2;
        const int local_tile_idx = i / 2;

        if (local_tile_idx + tile_idx_start < NUM_TILES)
        {
            float in_row[4];
            for (int col_idx = 0; col_idx < 4; ++col_idx)
                in_row[col_idx] = shared_2x4[local_tile_idx][row_idx][col_idx];

            float out_row[2];
            multiply_AT_2x2_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 2; ++col_idx)
                shared_4x4[local_tile_idx][row_idx * 2 + col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 2 * 2; i += BLOCK_SIZE)
    {
        const int offset = i % 4;
        const int within_tile_col_idx = offset % 2;
        const int within_tile_row_idx = offset / 2;

        const int local_tile_idx = i / 4;

        const int global_tile_idx = local_tile_idx + tile_idx_start;
        const int global_tile_col_idx = global_tile_idx % TILES_W;
        const int global_tile_row_idx = global_tile_idx / TILES_W;

        const int out_col = global_tile_col_idx * 2 + within_tile_col_idx;
        const int out_row = global_tile_row_idx * 2 + within_tile_row_idx;

        if (global_tile_idx < NUM_TILES && out_col < OUT_W && out_row < OUT_H)
            out[((n * K + k) * OUT_H + out_row) * OUT_W + out_col] = shared_4x4[local_tile_idx][offset];
    }
}
#endif