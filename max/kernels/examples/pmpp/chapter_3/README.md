# Chapter 3: Multidimensional grids and data

Moving from 1D to 2D, these examples use image processing and matrix
multiplication to demonstrate 2D grids and data. Each kernel maps a thread to a
pixel or matrix element using 2D index calculations.

## Files

| File | Description |
|------|-------------|
| `fig3_4.mojo` | Color-to-grayscale conversion; each thread converts one RGB pixel to a luminance value |
| `fig3_8.mojo` | Box blur; each thread computes the average of a pixel's neighborhood |
| `fig3_11.mojo` | Basic matrix multiplication; each thread computes one element of the output matrix |
