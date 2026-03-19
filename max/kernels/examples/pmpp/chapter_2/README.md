# Chapter 2: Heterogeneous data parallel computing

This chapter introduces the GPU programming model and walks through the first GPU program: vector addition. The examples build from a CPU baseline to a complete GPU implementation with memory management and kernel launch.

## Files

| File | Description |
|------|-------------|
| `fig2_4.mojo` | CPU-only vector addition, the sequential baseline before any GPU code |
| `fig2_8.mojo` | GPU memory management: allocate device buffers, copy host-to-device and device-to-host, free |
| `fig2_10.mojo` | Vector addition GPU kernel; each thread computes one element of C = A + B |
| `fig2_12.mojo` | Launch configuration: calculating grid and block dimensions with `ceildiv()` |
| `fig2_13.mojo` | Complete host function combining memory management and kernel launch |
| `vec_add.mojo` | Full working example: kernel + host function + correctness verification |

## Mojo vs. CUDA

`DeviceBuffer` and `DeviceContext` replace `cudaMalloc()`, `cudaMemcpy()`, and `cudaFree()`. The kernel itself is nearly identical; `global_idx()` maps to `blockIdx.x * blockDim.x + threadIdx.x`.

Start with `vec_add.mojo` for a runnable end-to-end example for this chapter.
