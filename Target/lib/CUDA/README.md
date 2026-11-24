# CUDA hardware support

This folder hosts the implementation of the Driver-HAL for CUDA.
We implement the C API functions in `SDK/.../HAL/M_*.h` in terms of
calls to CUDA driver API.

## Developer Notes

1. This plugin translates HAL API calls to CUDA driver calls.
2. It is a shared object that the Driver-HAL will `dlopen`/`dlsym`/`dlclose`.
3. It uses NVML (`libnvidia-ml.so.1`) for lightweight device enumeration and
   defers loading `libcuda.so.1` until a device handle or context is actually
   needed. Both libraries are loaded at runtime via `dlopen`+`dlsym` through
   global function pointers declared in `NVML.h` and `CUDALoader.h`.
   The official `cuda.h` and `nvml.h` headers are included at compile time
   for canonical type definitions and function declarations, but the shared
   libraries are never linked — they are loaded purely at runtime via `dlopen`.
4. The plugin shared object name is not given the `lib` prefix:
   plugins are not libraries in the traditional sense. To this end, our bazel
   target is a `binary` and not a `library` and the nomenclature includes an
   explicit `.so` as suffix.
