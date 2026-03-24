# Vendor CCL (RCCL/NCCL) integration

This package provides vendor collective communication (CCL) bindings for Mojo
kernels, supporting both AMD RCCL and NVIDIA NCCL at runtime. It is analogous
in scope to the vendor BLAS integration (e.g., cuBLAS) but focused on
collectives like all‑reduce and all‑gather.

## Overview

- Dynamic loading: the shared library is selected at runtime and loaded via a
  single resolver.
  - AMD: `librccl.so` (searched across common ROCm install paths)
  - NVIDIA: `libnccl.so` (searched across common distro/CUDA paths)
- Backend selection: chosen automatically based on
  `has_amd_gpu_accelerator()`.
- Stream handling: accepts a Mojo `DeviceContext` stream and converts it to the
  vendor stream type; the CCL API receives an `OpaquePointer` for the stream
  parameter.
- Exposed ops: `allreduce`, `allgather` wrappers dispatch to the appropriate
  vendor.
- Availability: lightweight symbol checks allow skipping vendor paths when the
  library is not present.

## Files

- `ccl.mojo`: core FFI and wrappers for vendor collectives.
  - `_init_ccl_dylib`: unified RCCL/NCCL loader
  - `_get_ccl_function`: symbol resolver from the unified handle
  - `_ccl_stream_ptr`: converts `DeviceContext` stream to vendor stream and
    returns `OpaquePointer`
  - Public wrappers: `allreduce`, `allgather`

## Dynamic loading

At startup of the first call, `ccl.mojo` resolves the vendor library:

- AMD paths (`RCCL_LIBRARY_PATHS`):
  - `librccl.so`, `librccl.so.1`, `/opt/rocm/lib/librccl.so*`
- NVIDIA paths (`NCCL_LIBRARY_PATHS`):
  - `libnccl.so`, `libnccl.so.2`, common distro/CUDA locations

Resolution uses `_find_dylib` with a per‑vendor search list. Errors are
surfaced as Mojo exceptions if resolution fails during use; availability
helpers allow prechecking.

## API

The module exposes higher‑level Mojo wrappers that normalize vendor
differences:

- `allreduce[dtype, rank, ngpus](inputs, outputs, list_of_ctx)`
  - Collective sum reduction over all ranks, writing results to each rank’s output.
- `allgather[dtype, rank, ngpus](inputs, outputs, list_of_ctx)`
  - Gathers all ranks’ inputs to a per‑rank output layout (equal per‑rank
    counts required).

Both APIs:

- Validate input sizes and `DeviceContext` count.
- Use `_ccl_stream_ptr(ctx)` to pass the correct vendor stream to the CCL call.
- Throw Mojo `Error` on vendor error codes.

## Availability checks

- `is_allreduce_available()` and `is_allgather_available()` perform
  best‑effort symbol resolution via `_get_ccl_function` without calling the
  function. Use these to skip vendor paths (e.g., on CI or machines without
  RCCL/NCCL).

## Notes & limitations

- Mixed‑vendor hosts are not explicitly supported; backend selection is automatic.
- Supported dtypes: `float32`, `bfloat16`, `float16` (others will raise an error).
- `allgather` requires equal element counts across ranks.
- Error messages and naming are vendor‑agnostic ("CCL").

## Contributing

- Keep new CCL ops following the same pattern:
  1. Add a function signature alias (if helpful).
  2. Resolve with `_get_ccl_function["symbol", Sig]()`.
  3. Use `_ccl_stream_ptr(ctx)` for the stream argument.
  4. Surface status via `_check_ccl_ok`.

- Prefer unified wrappers over duplicating RCCL/NCCL variants when signatures
  match.
