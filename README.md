# Non-Fused Winograd Convolution in CUDA C++
This project implements non-fused Winograd convolution for two configurations: 4x4 with 3x3 filters and 2x2 with 3x3 filters, where the first refers to the output tile size. The implementation is done in CUDA C++ and includes support for padding. Stride is not supported in this version. The project also includes autotuning to optimize hyperparameters for performance.
## Features
* <b>Winograd Convolution</b>: Implements Winograd convolution for 4x4 and 2x2 output tile sizes with 3x3 filters.
* <b>Padding Support</b>: Includes support for padding the input.
* <b>4 Phases of Computation</b>:
    1. <b>Filter Transform</b>
    2. <b>Input Transform</b>
    3. <b>Batched GEMM for Hadamard Product</b>
    4. <b>Inverse Transform</b>

## Comparison Against cuDNN
See comparison at [BENCHMARKS.md](BENCHMARKS.md)
## Usage
While the 4x4 approach is faster, it loses precision, with difference increasing with increasing parameters.

Nvidia cuDNN's winograd somehow has higher precision (we think they also use a 4x4_3x3 implementation).

Although convolution does not demand such high precision, but if you need high high precision use 2x2 implementation.

To use this implementation, follow the syntax and steps outlined below:
### Syntax
```cpp
#include "winograd_4x4_3x3.cuh"
#include "winograd_2x2_3x3.cuh"

const int N = 32,
          C = 128,
          H = 112,
          W = 112,
          K = 128,
          padding = 1;

float *img = new float[N * C * H * W];
float *filter = new float[K * C * 3 * 3];

float *out = convWinograd_2x2_3x3(img, N, C, H, W, f, K, padding); 
float *out = convWinograd_4x4_3x3(img, N, C, H, W, f, K, padding);
// out [N, K, H, W]

free(img); free(filter); free(out);
```
## Phases
Implementation can be seen at [PHASES.md](PHASES.md)

## Autotuning
We auto tuned the launch configuration for our hardware (RTX 3070Ti). Other devices may need another round of autotuning to get comparable performance.

## Building and Running
To build the project, ensure you have CUDA installed and properly configured. Use CMake to build the project.
```bash
mkdir build
cd build 
cmake ..
```
Run binary (main) from bin folder.

## Acknowledgements
* NVIDIA for their CUDA framework.
* <a href = "https://arxiv.org/abs/1509.09308">Fast Algorithms for Convolutional Neural Networks</a>