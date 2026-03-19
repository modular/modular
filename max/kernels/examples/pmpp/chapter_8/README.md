# Chapter 8: Stencil

This chapter applies the tiling and coarsening techniques from earlier chapters to 3D stencil computations, a pattern that appears in finite difference solvers, fluid simulation, and scientific computing.

## Files

| File | Description |
|------|-------------|
| `fig8_6.mojo` | Basic 3D stencil kernel; 7-point stencil (center + 6 face neighbors), loads directly from global memory |
| `fig8_8.mojo` | Tiled 3D stencil with shared memory; threads load an input tile including halo cells, reducing global memory reads for interior elements |
| `fig8_10.mojo` | Thread coarsening in the z-direction; each thread processes multiple z-layers, reusing the xy-plane data loaded into shared memory |
| `fig8_12.mojo` | Register tiling; keeps the current z-plane in registers and advances through z, reducing shared memory pressure further |

## Mojo vs. CUDA

Shared memory allocation and barrier usage follow the same pattern as Chapter 5. `FuncAttribute` controls the shared memory size at launch.

Read in order: `fig8_6.mojo` -> `fig8_8.mojo` -> `fig8_10.mojo` -> `fig8_12.mojo`.
