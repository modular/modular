# Chapter 19: Convolutional layers

A convolutional layer's forward pass can be reformulated as matrix
multiplication via im2col. The examples start with CPU reference implementations
and build up to a GPU kernel that performs this transformation on the fly,
without materializing the unrolled input.

## Files

| File | Description |
|------|-------------|
| `conv_utils.mojo` | Shared utilities; 4D tensor indexing helpers, CPU reference convolution, data initialization and verification |
| `fig19_3.mojo` | CPU forward pass, single image; `X[C, H, W]`, `F[M, C, K, K]` -> `Y[M, H_out, W_out]` |
| `fig19_4.mojo` | CPU forward pass, batched; `X[N, C, H, W]` -> `Y[N, M, H_out, W_out]` |
| `fig19_07.mojo` | GPU forward pass kernel; each thread computes one output pixel in one output feature map, direct convolution |
| `fig19_11.mojo` | Tiled im2col-based convolution; reformulates convolution as `Y = F * X_unrolled` and performs tiled matrix multiplication on the fly without materializing the unrolled input |

## Notes

`fig19_11.mojo` computes the im2col transformation inline as threads load data,
avoiding the memory cost of explicitly unrolling `X` first.

`conv_utils.mojo` is a dependency for all figure files in this directory.
