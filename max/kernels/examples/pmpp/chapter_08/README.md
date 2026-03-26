# Chapter 8: Stencil

Finite difference solvers, fluid simulations, and scientific codes all rely on
3D stencil computations. These examples apply tiling and coarsening from earlier
chapters to that problem.

## Files

| File           | Description                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `fig8_6.mojo`  | Basic 3D stencil kernel; 7-point stencil (center + 6 face neighbors), loads directly from global memory                                  |
| `fig8_8.mojo`  | Tiled 3D stencil with shared memory; threads load an input tile including halo cells, reducing global memory reads for interior elements |
| `fig8_10.mojo` | Thread coarsening in the z-direction; each thread processes multiple z-layers, reusing the xy-plane data loaded into shared memory       |
| `fig8_12.mojo` | Register tiling; keeps the current z-plane in registers and advances through z, reducing shared memory pressure further                  |
