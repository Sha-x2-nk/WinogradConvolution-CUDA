#ifndef WINOGRAD_4x4_3x3_ATtA_CUH
#define WINOGRAD_4x4_3x3_ATtA_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ void multiply_AT_4x4_3x3(const float in[6], float out[4])
{
    //  A = {
    //     1,  1,  1,  1,  1,  0,
    //     0,  1, -1,  2, -2,  0,
    //     0,  1,  1,  4,  4,  0,
    //     0,  1, -1,  8, -8,  1
    // };
    float temp1 = in[1] + in[2];
    float temp2 = in[1] - in[2];
    float temp3 = in[3] + in[4];
    float temp4 = in[3] - in[4];

    out[0] = in[0] + temp1 + temp3;
    out[1] = temp2 + 2 * temp4;
    out[2] = temp1 + 4 * temp3;
    out[3] = temp2 + 8 * temp4 + in[5];
}

template <int NUM_TILES_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd_4x4_3x3_ATtA(
    const float *t, const int N, const int K, const int TILES_H, const int TILES_W,
    float *out, const int OUT_H, const int OUT_W)
{
    /*
        input -> hadamard, [6 x 6 x K x N x TILES_H x TILES_W]
        output -> conv output, [N x K x OUT_H x OUT_W]
    */
    const int NUM_TILES = TILES_H * TILES_W;

    const int tile_idx_start = blockIdx.x * NUM_TILES_PER_BLOCK;
    const int k = blockIdx.y;
    const int n = blockIdx.z;

    __shared__ float shared_6x6[NUM_TILES_PER_BLOCK][6 * 6];

    for (int i = threadIdx.x; i < 6 * 6 * NUM_TILES_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int local_tile_idx = i % NUM_TILES_PER_BLOCK;
        const int global_tile_idx = local_tile_idx + tile_idx_start;

        const int offset = i / NUM_TILES_PER_BLOCK;

        if (global_tile_idx < NUM_TILES)
            shared_6x6[local_tile_idx][offset] = t[((offset * K + k) * N + n) * NUM_TILES + global_tile_idx];
    }
    __syncthreads();

    // computing ATt
    __shared__ float shared_4x6[NUM_TILES_PER_BLOCK][4][6];
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // take out a col
        const int col_idx = i % 6;
        const int tile_idx = i / 6;

        if (tile_idx_start + tile_idx < NUM_TILES)
        {
            float in_col[6];

            for (int row_idx = 0; row_idx < 6; ++row_idx)
                in_col[row_idx] = shared_6x6[tile_idx][row_idx * 6 + col_idx];

            float out_col[4];
            multiply_AT_4x4_3x3(in_col, out_col);

            for (int row_idx = 0; row_idx < 4; ++row_idx)
                shared_4x6[tile_idx][row_idx][col_idx] = out_col[row_idx];
        }
    }
    __syncthreads();

    // will now compute ATtA
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        // take out a row
        const int row_idx = i % 4;
        const int tile_idx = i / 4;

        if (tile_idx + tile_idx_start < NUM_TILES)
        {
            float in_row[6];
            for (int col_idx = 0; col_idx < 6; ++col_idx)
                in_row[col_idx] = shared_4x6[tile_idx][row_idx][col_idx];

            float out_row[4];
            multiply_AT_4x4_3x3(in_row, out_row);

            for (int col_idx = 0; col_idx < 4; ++col_idx)
                shared_6x6[tile_idx][row_idx * 4 + col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4 * 4; i += BLOCK_SIZE)
    {
        const int tile_offset = i % 16;
        const int within_tile_col_idx = tile_offset % 4;
        const int within_tile_row_idx = tile_offset / 4;

        const int local_tile_idx = i / 16;
        const int global_tile_idx = tile_idx_start + local_tile_idx;

        const int global_tile_col_idx = global_tile_idx % TILES_W;
        const int global_tile_row_idx = global_tile_idx / TILES_W;

        const int map_x = global_tile_col_idx * 4 + within_tile_col_idx;
        const int map_y = global_tile_row_idx * 4 + within_tile_row_idx;

        if (global_tile_idx < NUM_TILES && map_x < OUT_W && map_y < OUT_H)
            out[((n * K + k) * OUT_H + map_y) * OUT_W + map_x] = shared_6x6[local_tile_idx][tile_offset];
    }
}

#endif