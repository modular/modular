# Chapter 21: Electrostatic potential map

Direct Coulomb Summation (DCS) computes the electrostatic potential at every
point on a 3D grid from a set of charged atoms. Because every atom contributes
to every grid point, it is a natural all-pairs problem that appears in molecular
dynamics and computational chemistry. The examples use it to demonstrate thread
coarsening and the scatter vs. gather tradeoff.

## Files

| File             | Description                                                                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `dcs_utils.mojo` | Shared utilities; grid dimension struct, atom initialization, and verification functions                                                          |
| `fig21_03.mojo`  | CPU baseline (unoptimized); straightforward triple-nested loop: for each grid point, iterate over all atoms                                       |
| `fig21_04.mojo`  | CPU baseline (optimized); reorders the loops to hoist z-related calculations out of the inner loops, reducing redundant work                      |
| `fig21_05.mojo`  | GPU scatter kernel; each thread handles one atom and adds its contribution to all grid points, requires atomic operations                         |
| `fig21_06.mojo`  | GPU gather kernel; each thread handles one grid point and accumulates contributions from all atoms, no atomics needed                             |
| `fig21_08.mojo`  | GPU gather with thread coarsening; each thread handles multiple consecutive grid points (`COARSEN_FACTOR`), reusing atom data across calculations |

## Notes

The scatter pattern (`fig21_05.mojo`) requires atomic operations and has limited
parallelism when atom count is low. The gather pattern (`fig21_06.mojo`) is the
practical choice: one thread per grid point, no atomics. Thread coarsening in
`fig21_08.mojo` improves efficiency further.

**Note:** `fig21_10` (coalesced DCS kernel) has no Mojo port. Refer to the book
for the CUDA version.

`dcs_utils.mojo` is a dependency for all figure files in this directory.
