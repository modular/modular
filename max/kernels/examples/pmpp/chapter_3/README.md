# Chapter 3: Multidimensional grids and data

This chapter extends the GPU programming model to 2D grids and data, using image processing and matrix multiplication as the main examples. Each kernel maps a thread to a pixel or matrix element using 2D index calculations.

## Files

| File | Description |
|------|-------------|
| `fig3_4.mojo` | Color-to-grayscale conversion; each thread converts one RGB pixel to a luminance value |
| `fig3_8.mojo` | Box blur; each thread computes the average of a pixel's neighborhood |
| `fig3_11.mojo` | Basic matrix multiplication; each thread computes one element of the output matrix |

## Mojo vs. CUDA

`global_idx()` returns a struct with `.x` and `.y` components, replacing the 2D index arithmetic from `blockIdx` and `threadIdx` in CUDA. Boundary checking is the same.
