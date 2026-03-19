# Chapter 19: Convolutional layers

This chapter covers GPU implementation of the forward pass of a convolutional layer. The examples start with CPU reference implementations and build up to an optimized GPU kernel that reformulates convolution as matrix multiplication (im2col).

## Files

| File | Description |
|------|-------------|
| `conv_utils.mojo` | Shared utilities; 4D tensor indexing helpers, CPU reference convolution, data initialization and verification |
| `fig19_3.mojo` | CPU forward pass, single image; `X[C, H, W]`, `F[M, C, K, K]` -> `Y[M, H_out, W_out]` |
| `fig19_4.mojo` | CPU forward pass, batched; `X[N, C, H, W]` -> `Y[N, M, H_out, W_out]` |
| `fig19_07.mojo` | GPU forward pass kernel; each thread computes one output pixel in one output feature map, direct convolution |
| `fig19_11.mojo` | Tiled im2col-based convolution; reformulates convolution as `Y = F * X_unrolled` and performs tiled matrix multiplication on the fly without materializing the unrolled input |

## Notes

`fig19_11.mojo` computes the im2col transformation inline as threads load data, avoiding the memory cost of explicitly unrolling `X` first.

`conv_utils.mojo` must be in the same directory as the figure files when compiling.
