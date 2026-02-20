# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.

# GPU Target Configuration Guide

When adding support for a new GPU architecture, you must create a target
configuration function that returns a `_TargetType`. This guide explains the
components of the MLIR target configuration, with special focus on the
`data_layout` string.

## MLIR Target Components

Each GPU target function returns an MLIR `kgen.target` attribute with these
fields:

- **triple**: Target triple (e.g., "nvptx64-nvidia-cuda", "amdgcn-amd-amdhsa").
- **arch**: Architecture name (e.g., "sm_80", "gfx942", "apple-m4").
- **features**: Target-specific features (e.g., "+ptx81,+sm_80").
- **tune_cpu**: Optimization target (usually same as arch, can differ for
  tuning).
- **data_layout**: LLVM data layout string (explained in detail below).
- **index_bit_width**: Bit width for index types (usually 64).
- **simd_bit_width**: SIMD register width (usually 128 for modern GPUs).

## Understanding Data Layout Strings

The `data_layout` string describes memory layout characteristics for the target
architecture. It follows LLVM' data layout specification format: https://llvm.org/docs/LangRef.html#data-layout
and is used by the compiler to make decisions about memory access patterns,
type layouts, and optimizations.

### Format Overview

The string consists of specifications separated by dashes (`-`):

- **Endianness**: `e` (little-endian) or `E` (big-endian).
- **Pointers**: `p[addr_space]:size:abi:pref:idx`.
- **Integers**: `i<size>:<abi>:<pref>`.
- **Floats**: `f<size>:<abi>:<pref>`.
- **Vectors**: `v<size>:<abi>:<pref>`.
- **Native widths**: `n<size>:<size>:...`.
- **Stack alignment**: `S<size>`.
- **Address space**: `A<number>`.
- **Mangling**: `m:<style>` (e.g., `m:e` for ELF).

### Component Details

#### Endianness

- `e`: Little-endian (all modern GPUs use this).
- `E`: Big-endian (rarely used).

#### Pointer Specifications: `p[addr_space]:size:abi:pref:idx`

Defines pointer sizes and alignments for different memory spaces:

- **Address space**: Optional number (0-9) specifying memory type:
  - `p` or `p0`: Generic/flat address space.
  - `p1`: Global memory (AMD) or device memory.
  - `p2`: Constant memory (AMD).
  - `p3`: Shared/local memory (NVIDIA) or local memory (AMD).
  - `p4`: Constant memory (NVIDIA) or generic memory (AMD).
  - `p5`: Local/private memory (NVIDIA/AMD).
  - `p6-p9`: Vendor-specific address spaces.
- **size**: Pointer size in bits.
- **abi**: ABI-required alignment in bits.
- **pref**: Preferred alignment in bits (optional).
- **idx**: Index type size in bits (optional).

Examples:

- `p3:32:32` means shared memory uses 32-bit pointers with 32-bit alignment.
- `p:64:64:64` means generic pointers are 64 bits with 64-bit alignment.
- `p7:160:256:256:32` means address space 7 uses 160-bit pointers with 256-bit
  alignment.

#### Integer Specifications: `i<size>:<abi>:<pref>`

Defines alignment for integer types:

- **size**: Integer size in bits (1, 8, 16, 32, 64, 128, 256, etc.).
- **abi**: Minimum ABI alignment in bits.
- **pref**: Preferred alignment in bits (optional, defaults to abi).

Examples:

- `i64:64` means 64-bit integers have 64-bit alignment.
- `i128:128` means 128-bit integers have 128-bit alignment.
- `i1:8:8` means 1-bit booleans are stored in 8-bit aligned bytes.

#### Float Specifications: `f<size>:<abi>:<pref>`

Similar to integers but for floating-point types:

Examples:

- `f32:32:32` means 32-bit floats have 32-bit alignment.
- `f64:64:64` means 64-bit doubles have 64-bit alignment.

#### Vector Specifications: `v<size>:<abi>:<pref>`

Defines alignment for vector types:

- **size**: Vector size in bits.
- **abi**: ABI alignment in bits.
- **pref**: Preferred alignment in bits (optional).

Examples:

- `v16:16` means 16-bit vectors aligned to 16 bits.
- `v128:128:128` means 128-bit vectors have 128-bit alignment.

#### Native Integer Widths: `n<size>:<size>:...`

Specifies which integer widths are "native" (efficient) for the target. The
compiler will prefer these sizes for operations.

Examples:

- `n16:32:64` means 16, 32, and 64-bit operations are efficient.
- `n32:64` means 32 and 64-bit operations are efficient.
- `n8:16:32` means 8, 16, and 32-bit operations are efficient.

#### Stack Alignment: `S<size>`

Specifies natural stack alignment in bits.

Example: `S32` means 32-bit stack alignment.

#### Address Space: `A<number>`

Specifies the default address space for allocations.

Example: `A5` means use address space 5 by default.

#### Non-Integral Pointers: `ni:<space>:<space>:...`

Lists address spaces where pointers cannot be cast to integers.

Example: `ni:7:8:9` means address spaces 7, 8, and 9 have non-integral pointers.

## Vendor-Specific Patterns

### NVIDIA GPUs (CUDA/PTX)

Typical data layout for NVIDIA GPUs (sm_60 and later):

```
e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64
```

Breakdown:

- `e`: Little-endian.
- `p3:32:32`: Shared memory pointers are 32-bit.
- `p4:32:32`: Constant memory pointers are 32-bit.
- `p5:32:32`: Local memory pointers are 32-bit.
- `p6:32:32`, `p7:32:32`: NVIDIA-specific address spaces.
- `i64:64`, `i128:128`, `i256:256`: Integer alignments.
- `v16:16`, `v32:32`: Vector alignments for warp operations.
- `n16:32:64`: Native integer widths (16, 32, and 64-bit operations).

Note: NVIDIA GPUs use address-space-specific 32-bit pointers for shared,
constant, and local memory, while the default address space (not specified)
uses 64-bit pointers. This matches the PTX memory model.

### AMD GPUs (ROCm/HIP)

Typical data layout for AMD GPUs (CDNA and RDNA):

```
e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9
```

AMD GPUs use more address spaces and have more complex specifications:

- `m:e`: ELF mangling style.
- `p:64:64`: Default pointers are 64-bit (unified addressing).
- `p1:64:64`: Global memory uses 64-bit pointers.
- `p2:32:32`: Constant memory uses 32-bit pointers.
- `p3:32:32`: Local/shared memory uses 32-bit pointers.
- `p4:64:64`: Generic address space uses 64-bit pointers.
- `p5:32:32`: Private memory uses 32-bit pointers.
- `p7`, `p8`, `p9`: Complex buffer descriptors (160, 128, 192 bits).
- Extensive vector sizes (`v16` through `v2048`) for wavefront operations.
- `n32:64`: Native integer widths.
- `S32`: 32-bit stack alignment.
- `A5`: Default address space is 5.
- `G1`: Global address space is 1.
- `ni:7:8:9`: Address spaces 7, 8, 9 have non-integral pointers.

### Apple Metal GPUs

Typical data layout for Apple Silicon:

```
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32
```

Apple GPUs have unified memory architecture:

- `p:64:64:64`: 64-bit pointers with explicit preferred alignment (unified
  memory).
- Explicit specifications for all integer sizes (`i1`, `i8`, `i16`, `i32`,
  `i64`).
- Explicit float alignments (`f32:32:32`, `f64:64:64`).
- Comprehensive vector size coverage (`v16` through `v1024`).
- `n8:16:32`: Native integer widths (8, 16, and 32-bit operations).

## How to Obtain Data Layout Strings

When adding support for a new GPU architecture, obtain the data layout string
using these methods:

### Method 1: Query LLVM/Clang (Recommended)

Use Clang to query the target's default data layout:

For NVIDIA GPUs:

```bash
echo 'target triple = "nvptx64-nvidia-cuda"' > test.ll
clang -S test.ll -o - | grep datalayout
```

For AMD GPUs:

```bash
echo 'target triple = "amdgcn-amd-amdhsa"' > test.ll
clang -S test.ll -o - | grep datalayout
```

### Method 2: Consult LLVM Source Code

Check the LLVM source for target data layout definitions:

- **NVIDIA**: `llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp` (see
  `computeDataLayout()`).
- **AMD**: `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` (see
  `getGPUDataLayout()`).

### Method 3: Reference Similar GPUs

For GPUs in the same architecture family, the data layout is often identical:

- All NVIDIA Ampere/Ada/Hopper GPUs (sm_80+) use the same data layout.
- AMD CDNA GPUs share similar layouts.
- Apple Metal GPUs have consistent patterns across generations.

When in doubt, use the data layout from a GPU in the same family.

### Method 4: Consult Vendor Documentation

Refer to official programming guides and specifications:

- **NVIDIA**: [LLVM NVPTX Usage Guide](https://llvm.org/docs/NVPTXUsage.html#data-layout),
  CUDA Programming Guide, PTX ISA documentation.
- **AMD**: ROCm documentation, LLVM AMDGPU documentation.
- **Apple**: Metal Programming Guide, Metal Shading Language Specification.

The LLVM NVPTX documentation recommends this data layout for 64-bit NVIDIA GPUs:

```
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64
```

Note: The data layouts in this file use address-space-specific pointer
specifications (p3, p4, p5, etc.) rather than the generic `p:64:64:64`. This
provides more precise control over memory access patterns for different memory
spaces in GPU kernels.

## Field-by-Field Explanation

### Triple

The target triple identifies the architecture, vendor, and operating system:

- **NVIDIA**: `nvptx64-nvidia-cuda` (64-bit) or `nvptx-nvidia-cuda` (32-bit).
- **AMD**: `amdgcn-amd-amdhsa` (HSA runtime).
- **Apple**: `air64-apple-macosx` (Metal on macOS).

### Arch

The architecture name specifies the GPU generation:

- **NVIDIA**: `sm_XX` where XX is the compute capability (e.g., `sm_80` for
  compute 8.0).
  - Find compute capability at https://developer.nvidia.com/cuda-gpus.
  - Format: `sm_XY` maps to compute capability `X.Y`, `sm_XYZ` maps to `XY.Z`.
- **AMD**: `gfxXXXX` where XXXX is the GFX version (e.g., `gfx942` for MI300X).
  - Find GFX version in ROCm documentation or GPU specifications.
- **Apple**: `apple-mX` where X is the chip generation (e.g., `apple-m4`).

### Features

Target-specific features enabled for code generation:

- **NVIDIA**: `+ptxXX,+sm_YY` where XX is PTX version and YY is compute
  capability.
  - PTX version should match your CUDA toolkit version (see PTX ISA docs).
  - Example: `+ptx85,+sm_90a` enables PTX 8.5 and compute 9.0a features.
  - **Q: Is specifying PTX version redundant?** A: No, PTX version determines
    available instructions and features, independent of compute capability.
- **AMD**: Often empty (`""`) as features are implied by architecture.
- **Apple**: Often empty (`""`) for Metal GPUs.

### Tune CPU

Specifies the optimization target for code generation:

- Usually the same as `arch` (e.g., `tune_cpu = "sm_90a"`).
- Can differ if you want to optimize for a different microarchitecture while
  maintaining compatibility (e.g., `arch = "sm_80"`, `tune_cpu = "sm_90a"`).
- Some older GPU entries omit this field (see GTX 970, GTX 1080 Ti).

### Index Bit Width

The bit width for index types used in address calculations:

- **32-bit systems**: `index_bit_width = 32`.
- **64-bit systems**: `index_bit_width = 64`.
- Most modern GPUs use 64-bit indexing for large memory spaces.

### SIMD Bit Width

The width of SIMD registers in bits:

- **Modern GPUs**: Usually `simd_bit_width = 128` (128-bit vector operations).
- This represents the native vector width for efficient operations.
- **How to find this**: Based on warp/wavefront width and register
  architecture:
  - NVIDIA: 128 bits (4 x 32-bit values per warp operation).
  - AMD: 128 bits for CDNA/RDNA architectures.
  - Apple: 128 bits for Metal GPUs.

## Step-by-Step Guide for Adding a New GPU

Follow these steps to add support for a new GPU architecture:

### Step 1: Gather GPU Information

Collect these specifications for your GPU:

- **Model name**: e.g., "H100", "MI300X", "M4".
- **Compute capability** (NVIDIA) or **GFX version** (AMD) or **Metal version**
  (Apple).
- **Architecture family**: Identify the family (e.g., Hopper, CDNA3, Apple M
  series).
- **SM/CU count**: Number of streaming multiprocessors or compute units.
- **Target triple**: Standard LLVM triple for the vendor.
- **Data layout string**: Obtain using methods described above.

To find SM count for NVIDIA GPUs, use this CUDA code:

```c
void printMultiProcessorCount() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Number of SMs: %d\\n", deviceProp.multiProcessorCount);
}
```

Or check vendor specifications:

- **NVIDIA**: https://developer.nvidia.com/cuda-gpus.
- **AMD**: ROCm device specifications.

### Step 2: Create the Target Function

Add a new function that returns the MLIR target configuration.

Example for NVIDIA GPU:

```mojo
fn _get_your_gpu_target() -> _TargetType:
    \"\"\"Creates an MLIR target configuration for Your GPU.

    Returns:
        MLIR target configuration for Your GPU.
    \"\"\"
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
```

Place this function with other GPU target functions in this file (search for
`_get_*_target()` functions).

### Step 3: Create the GPUInfo Alias

Define the GPU characteristics using the appropriate architecture family:

```mojo
comptime YourGPU = GPUInfo.from_family(
    family=NvidiaHopperFamily,  # Choose the appropriate family
    name="Your GPU",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="hopper",
    compute=9.0,  # Must match arch (9.0 -> sm_90, 12.1 -> sm_121)
    version="sm_90a",
    sm_count=132,  # Number of streaming multiprocessors
)
```

Place this alias with other GPU aliases in this file.

### Step 4: Update `_get_info_from_target`

Add your architecture to the constraint list in the `_get_info_from_target`
function:

```mojo
comptime assert StaticString(target_arch)
    in (
        # NVIDIA
        StaticString("cuda"),
        StaticString("52"),
        StaticString("90a"),  # Add your architecture here
        # ... rest of architectures ...
    ), String("the target architecture '",
    target_arch0,
    "' is invalid or not currently supported")
```

Then add the mapping in the `@parameter` block:

```mojo
comptime if target_arch == "52":
    return materialize[GTX970]()
elif target_arch == "90a":  # Add your mapping here
    return materialize[YourGPU]()
# ... rest of mappings ...
```

Note: The `target_arch` has the "sm_" prefix stripped, so "sm_90a" becomes
"90a".

Note: GPUs are currently 1:1 with the `target_arch` string. This is going to be
changed to support multiple GPUs per target_arch in the future.

### Step 5: Update `GPUInfo.target` Method

Add the target mapping in the `target()` method of the `GPUInfo` struct:

```mojo
fn target(self) -> _TargetType:
    \"\"\"Gets the MLIR target configuration for this GPU.

    Returns:
        MLIR target configuration for the GPU.
    \"\"\"
    if self.name == "NVIDIA Tesla P100":
        return _get_teslap100_target()
    if self.name == "Your GPU":  # Add your GPU here
        return _get_your_gpu_target()
    # ... rest of mappings ...
```

### Step 6: Build and Test

Build the standard library to verify your changes:

```bash
./bazelw build //mojo/stdlib/std
```

Test with a simple GPU program:

```bash
MODULAR_MOJO_MAX_IMPORT_PATH=bazel-bin/mojo/stdlib/std mojo your_test.mojo
```

Run existing GPU tests to ensure nothing broke:

```bash
./bazelw test //mojo/stdlib/test/gpu/...
```

## Common Pitfalls

Avoid these common mistakes when adding GPU support:

1. **Mismatched compute capability**: Ensure `compute` matches `arch` (e.g.,
   `compute=9.0` with `arch="sm_90a"`).
2. **Incorrect pointer sizes**: Verify address space pointer sizes match
   hardware capabilities.
3. **Missing vector alignments**: Include all vector sizes your kernels will
   use.
4. **Wrong endianness**: All modern GPUs are little-endian (use `e`).
5. **Inconsistent with LLVM**: Data layout must match LLVM's target definition.
6. **Copy-paste errors**: Double-check field values when adapting from similar
   GPUs.
7. **Forgetting to update all 5 locations**: Target function, alias, constraint
   list, parameter block, and target() method.
8. **PTX/driver version mismatch**: Ensure PTX version is supported by your
   CUDA driver.

## Validation Checklist

Before submitting your GPU addition:

- [ ] Target function created and documented.
- [ ] GPUInfo alias defined with correct family.
- [ ] Architecture added to constraint list in `_get_info_from_target`.
- [ ] Mapping added to `@parameter` block in `_get_info_from_target`.
- [ ] Mapping added to `GPUInfo.target()` method.
- [ ] Data layout string validated against LLVM documentation.
- [ ] Compute capability matches architecture name.
- [ ] SM/CU count verified against official specifications.
- [ ] Standard library builds successfully.
- [ ] Existing tests pass.
- [ ] Manual testing with simple GPU kernel.

## Related Files

- **sys/info.mojo**: Defines `_TargetType` as `!kgen.target` and
  `CompilationTarget` struct.
- **LLVM Documentation**: https://llvm.org/docs/LangRef.html#data-layout
  (complete data layout specification).
- **LLVM NVPTX Usage**: https://llvm.org/docs/NVPTXUsage.html (NVIDIA-specific
  guidance).

## Examples in This File

See real-world examples by searching for these functions:

- `_get_h100_target()`: NVIDIA Hopper H100 (compute 9.0).
- `_get_mi300x_target()`: AMD CDNA3 MI300X.
- `_get_metal_m4_target()`: Apple Metal M4.
- `_get_rtx5090_target()`: NVIDIA Blackwell consumer GPU.

Each example demonstrates the complete target configuration for that GPU family.
"""

from math import ceildiv, floor
from os import abort
from sys.info import CompilationTarget, _accelerator_arch, _TargetType

comptime _KB = 1024
comptime _K = 1024

# NVIDIA Architecture Families
comptime NvidiaMaxwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=96 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Maxwell architecture family (sm_50-sm_53)."""

comptime NvidiaPascalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Pascal architecture family (sm_60-sm_62)."""

comptime NvidiaTuringFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Turing architecture family (sm_75)."""

# Ampere architecture has three distinct variants based on compute capability:
# - sm_80: High-end datacenter (A100)
# - sm_86: Workstation/cloud (A10, RTX A-series)
# - sm_87: Embedded/edge (Jetson Orin)

comptime NvidiaAmpereDatacenterFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere datacenter architecture family (sm_80)."""

comptime NvidiaAmpereWorkstationFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere workstation architecture family (sm_86)."""

comptime NvidiaAmpereEmbeddedFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=164 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ampere embedded architecture family (sm_87)."""

comptime NvidiaAdaFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Ada Lovelace architecture family (sm_89)."""

comptime NvidiaHopperFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Hopper architecture family (sm_90)."""

comptime NvidiaBlackwellFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=228 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell datacenter architecture family (sm_100)."""

comptime NvidiaBlackwellConsumerFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=48 * 32,
    shared_memory_per_multiprocessor=100 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""NVIDIA Blackwell consumer architecture family (sm_120)."""

# AMD Architecture Families
comptime AMDCDNA3Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=64 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA3 architecture family (gfx94x)."""

comptime AMDCDNA4Family = AcceleratorArchitectureFamily(
    warp_size=64,
    threads_per_multiprocessor=64 * 32,
    shared_memory_per_multiprocessor=160 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""AMD CDNA4 architecture family (gfx95x)."""

comptime AMDRDNAFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=32 * _K,
    max_thread_block_size=_K,
)
"""AMD RDNA architecture family."""

# Apple Architecture Families
comptime AppleMetalFamily = AcceleratorArchitectureFamily(
    warp_size=32,
    threads_per_multiprocessor=32 * 32,
    shared_memory_per_multiprocessor=32 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""Apple Metal GPU architecture family."""

# ===-----------------------------------------------------------------------===#
# AcceleratorArchitectureFamily
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AcceleratorArchitectureFamily(TrivialRegisterPassable):
    """Defines common defaults for a GPU architecture family.

    This struct captures the shared characteristics across GPUs in the same
    architecture family, reducing redundancy when defining new GPU models.
    """

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""


# ===-----------------------------------------------------------------------===#
# Vendor
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Vendor(Equatable, TrivialRegisterPassable, Writable):
    """Represents GPU vendors.

    This struct provides identifiers for different GPU vendors and utility
    methods for comparison and string representation.

    The Vendor struct defines constants for common GPU vendors (NVIDIA, AMD)
    and includes a NO_GPU option for systems without GPU support. It provides
    comparison operators and string conversion methods for vendor identification.
    """

    var _value: Int8
    """The underlying integer value representing the vendor."""

    comptime NO_GPU = Self(0)
    """Represents no GPU or CPU-only execution."""

    comptime AMD_GPU = Self(1)
    """Represents AMD GPU vendor."""

    comptime NVIDIA_GPU = Self(2)
    """Represents NVIDIA GPU vendor."""

    comptime APPLE_GPU = Self(3)
    """Represents Apple GPU vendor."""

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are equal, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are not equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are not equal, False otherwise.
        """
        return not (self == other)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes vendor information to a writer.

        Args:
            writer: The writer to output vendor information to.
        """
        if self == Vendor.NO_GPU:
            writer.write("no_gpu")
            return
        if self == Vendor.AMD_GPU:
            writer.write("amd_gpu")
            return
        if self == Vendor.APPLE_GPU:
            writer.write("apple_gpu")
            return
        if self == Vendor.NVIDIA_GPU:
            writer.write("nvidia_gpu")
            return

        abort("unable to format unrecognized `Vendor` value")

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the vendor.

        Returns:
            String representation of the vendor.
        """
        return String.write(self)


# ===-----------------------------------------------------------------------===#
# NoGPU
# ===-----------------------------------------------------------------------===#


fn _get_empty_target() -> _TargetType:
    """Creates an empty target configuration for when no GPU is available.

    Returns:
        An empty MLIR target configuration.
    """
    return __mlir_attr[
        `#kgen.target<triple = "", `,
        `arch = "", `,
        `features = "", `,
        `data_layout="",`,
        `index_bit_width = 0,`,
        `simd_bit_width = 0`,
        `> : !kgen.target`,
    ]


comptime NoGPU = GPUInfo(
    name="NoGPU",
    vendor=Vendor.NO_GPU,
    api="none",
    arch_name="no_gpu",
    compute=0,
    version="",
    sm_count=0,
    warp_size=0,
    threads_per_multiprocessor=0,
    shared_memory_per_multiprocessor=0,
    max_registers_per_block=0,
    max_thread_block_size=0,
)
"""Placeholder for when no GPU is available."""


# ===-----------------------------------------------------------------------===#
# Apple Silicon
# ===-----------------------------------------------------------------------===#
fn _get_metal_m1_target() -> _TargetType:
    """Creates an MLIR target configuration for M1 Metal GPU.

    Returns:
        MLIR target configuration for M1 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m1", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m2_target() -> _TargetType:
    """Creates an MLIR target configuration for M2 Metal GPU.

    Returns:
        MLIR target configuration for M2 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m2", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m3_target() -> _TargetType:
    """Creates an MLIR target configuration for M3 Metal GPU.

    Returns:
        MLIR target configuration for M3 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m3", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m4_target() -> _TargetType:
    """Creates an MLIR target configuration for M4 Metal GPU.

    Returns:
        MLIR target configuration for M4 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m4", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_metal_m5_target() -> _TargetType:
    """Creates an MLIR target configuration for M5 Metal GPU.

    Returns:
        MLIR target configuration for M5 Metal.
    """
    return __mlir_attr[
        `#kgen.target<triple = "air64-apple-macosx", `,
        `arch = "apple-m5", `,
        `features = "", `,
        `data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MetalM1 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M1",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m1",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=8,  # M1 has 8 GPU cores
)
"""Apple M1 GPU configuration."""

comptime MetalM2 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M2",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m2",
    compute=3.0,  # Metal version 3.0
    version="metal_3",
    sm_count=10,  # M2 has 10 GPU cores
)
"""Apple M2 GPU configuration."""

comptime MetalM3 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M3",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m3",
    compute=3.0,  # Metal version 3.0 for M3
    version="metal_3",
    sm_count=10,  # M3 has 10 GPU cores
)
"""Apple M3 GPU configuration."""

comptime MetalM4 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M4",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m4",
    compute=4.0,  # Metal version 4.0 for M4
    version="metal_4",
    sm_count=10,  # M4 has 10 GPU cores
)
"""Apple M4 GPU configuration."""

comptime MetalM5 = GPUInfo.from_family(
    family=AppleMetalFamily,
    name="M5",
    vendor=Vendor.APPLE_GPU,
    api="metal",
    arch_name="apple-m5",
    compute=4.0,  # Metal version 4.0 for M5
    version="metal_4",
    sm_count=10,  # M5 has 10 GPU cores
)
"""Apple M5 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# A100
# ===-----------------------------------------------------------------------===#

# Note: features = "+ptx81" means that the kernel should be compiled using
# PTX version 8.1. This must be less than or equal to the installed CUDA
# driver's maximum supported PTX version. Currently we hardcode this to
# PTX version 8.1 which means that you need to have a CUDA driver included with
# CUDA 12.5 toolkit. The mapping from CUDA Driver to PTX version can be found by
# looking at the PTX ISA in the versioned docs
# https://developer.nvidia.com/cuda-toolkit-archive.


fn _get_a100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA A100 GPU.

    Returns:
        MLIR target configuration for A100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime A100 = GPUInfo.from_family(
    family=NvidiaAmpereDatacenterFamily,
    name="A100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.0,
    version="sm_80",
    sm_count=108,
)
"""NVIDIA A100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# A10
# ===-----------------------------------------------------------------------===#


fn _get_a10_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA A10 GPU.

    Returns:
        MLIR target configuration for A10.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime A10 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="A10",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=72,
)
"""NVIDIA A10 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Orin Nano
# ===-----------------------------------------------------------------------===#


fn _get_orin_nano_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Jetson Orin Nano GPU.

    Returns:
        MLIR target configuration for Orin Nano.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_87", `,
        `features = "+ptx81,+sm_87", `,
        `tune_cpu = "sm_87", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime OrinNano = GPUInfo.from_family(
    family=NvidiaAmpereEmbeddedFamily,
    name="Orin Nano",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.7,
    version="sm_87",
    sm_count=8,
)
"""NVIDIA Orin Nano GPU configuration."""

# ===-----------------------------------------------------------------------===#
# Jetson Thor
# ===-----------------------------------------------------------------------===#


fn _get_jetson_thor_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Jetson Thor.

    Returns:
        MLIR target configuration for Jetson Thor.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_110", `,
        `features = "+ptx85,+sm_110", `,
        `tune_cpu = "sm_110", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


comptime JetsonThor = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="Jetson Thor",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=11.0,
    version="sm_110",
    sm_count=20,
)
"""NVIDIA Jetson Thor GPU configuration."""

# ===-----------------------------------------------------------------------===#
# DGX Spark
# ===-----------------------------------------------------------------------===#


fn _get_dgx_spark_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA DGX Spark.

    Returns:
        MLIR target configuration for DGX Spark.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_121", `,
        `features = "+ptx86,+sm_121", `,
        `tune_cpu = "sm_121", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime DGXSpark = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="DGX Spark",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=12.1,
    version="sm_121",
    sm_count=48,
)
"""NVIDIA DGX Spark GPU configuration."""

# ===-----------------------------------------------------------------------===#
# L4
# ===-----------------------------------------------------------------------===#


fn _get_l4_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA L4 GPU.

    Returns:
        MLIR target configuration for L4.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_89", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime L4 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="L4",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada",
    compute=8.9,
    version="sm_89",
    sm_count=58,
)
"""NVIDIA L4 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090 M
# ===-----------------------------------------------------------------------===#


fn _get_rtx4090m_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 4090 Mobile GPU.

    Returns:
        MLIR target configuration for RTX 4090M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX4090m = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090m",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=76,
)
"""NVIDIA RTX 4090 Mobile GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX 4090
# ===-----------------------------------------------------------------------===#


fn _get_rtx4090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 4090.

    Returns:
        MLIR target configuration for RTX 4090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX4090 = GPUInfo.from_family(
    family=NvidiaAdaFamily,
    name="RTX4090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ada lovelace",
    compute=8.9,
    version="sm_89",
    sm_count=128,
)
"""NVIDIA RTX 4090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# H100
# ===-----------------------------------------------------------------------===#


fn _get_h100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA H100 GPU.

    Returns:
        MLIR target configuration for H100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
comptime H100 = GPUInfo.from_family(
    family=NvidiaHopperFamily,
    name="H100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="hopper",
    compute=9.0,
    version="sm_90a",
    sm_count=132,
)
"""NVIDIA H100 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# B100
# ===-----------------------------------------------------------------------===#


fn _get_b100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA B100 GPU.

    Returns:
        MLIR target configuration for B100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_100a", `,
        `features = "+ptx88,+sm_100a", `,
        `tune_cpu = "sm_100a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://resources.nvidia.com/en-us-blackwell-architecture
# TODO: Update once we have B100 access.
comptime B100 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=132,
)
"""NVIDIA B100 GPU configuration."""

comptime B200 = GPUInfo.from_family(
    family=NvidiaBlackwellFamily,
    name="B200",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=10.0,
    version="sm_100a",
    sm_count=148,
)
"""NVIDIA B200 GPU configuration."""

# ===-----------------------------------------------------------------------===#
# RTX5090
# ===-----------------------------------------------------------------------===#


fn _get_rtx5090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX5090 GPU.

    Returns:
        MLIR target configuration for RTX5090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_120a", `,
        `features = "+ptx86,+sm_120a", `,
        `tune_cpu = "sm_120a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
comptime RTX5090 = GPUInfo.from_family(
    family=NvidiaBlackwellConsumerFamily,
    name="RTX5090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="blackwell",
    compute=12.0,
    version="sm_120a",
    sm_count=170,
)
"""NVIDIA RTX 5090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX3090
# ===-----------------------------------------------------------------------===#


fn _get_rtx3090_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GeForce RTX 3090.

    Returns:
        MLIR target configuration for NVIDIA GeForce RTX 3090.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx63,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


# https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
comptime RTX3090 = GPUInfo.from_family(
    family=NvidiaAmpereWorkstationFamily,
    name="NVIDIA GeForce RTX 3090",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="ampere",
    compute=8.6,
    version="sm_86",
    sm_count=82,
)
"""NVIDIA GeForce RTX 3090 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1080Ti
# ===-----------------------------------------------------------------------===#


fn _get_gtx1080ti_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GTX 1080 Ti GPU.

    Returns:
        MLIR target configuration for GTX 1080 Ti.
    """
    # Note: GTX 1080 Ti doesn't specify tune_cpu, data_layout, or index_bit_width
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX1080Ti = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1080 Ti",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=28,
)
"""NVIDIA GeForce GTX 1080 Ti GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX1060
# ===-----------------------------------------------------------------------===#


fn _get_gtx1060_target() -> _TargetType:
    """
    Creates an MLIR target configuration for NVIDIA GTX 1060 GPU.

    Returns:
        MLIR target configuration for GTX 1060.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_61", `,
        `features = "+ptx50,+sm_61", `,
        `tune_cpu = "sm_61", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX1060 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA GeForce GTX 1060",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="pascal",
    compute=6.1,
    version="sm_61",
    sm_count=10,
)
"""NVIDIA GeForce GTX 1060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GTX970
# ===-----------------------------------------------------------------------===#


fn _get_gtx970_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA GTX 970 GPU.

    Returns:
        MLIR target configuration for GTX 970.
    """
    # Note: GTX 970 doesn't specify tune_cpu, data_layout, or index_bit_width
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_52", `,
        `features = "+ptx50,+sm_52", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime GTX970 = GPUInfo.from_family(
    family=NvidiaMaxwellFamily,
    name="NVIDIA GeForce GTX 970",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="maxwell",
    compute=5.2,
    version="sm_52",
    sm_count=13,
)
"""NVIDIA GeForce GTX 970 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Tesla P100
# ===-----------------------------------------------------------------------===#


fn _get_teslap100_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA Tesla P100 GPU.

    Returns:
        MLIR target configuration for Tesla P100.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_60", `,
        `features = "+ptx50,+sm_60", `,
        `tune_cpu = "sm_60", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime TeslaP100 = GPUInfo.from_family(
    family=NvidiaPascalFamily,
    name="NVIDIA Tesla P100",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="pascal",
    compute=6.0,
    version="sm_60",
    sm_count=56,
)
"""NVIDIA Tesla P100 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# RTX2060
# ===-----------------------------------------------------------------------===#


fn _get_rtx2060_target() -> _TargetType:
    """Creates an MLIR target configuration for NVIDIA RTX 2060 GPU.

    Returns:
        MLIR target configuration for RTX 2060.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_75", `,
        `features = "+ptx63,+sm_75", `,
        `tune_cpu = "sm_75", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime RTX2060 = GPUInfo.from_family(
    family=NvidiaTuringFamily,
    name="RTX2060",
    vendor=Vendor.NVIDIA_GPU,
    api="cuda",
    arch_name="turing",
    compute=7.5,
    version="sm_75",
    sm_count=30,
)
"""NVIDIA RTX 2060 GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI300X
# ===-----------------------------------------------------------------------===#


fn _get_mi300x_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI300X GPU.

    Returns:
        MLIR target configuration for MI300X.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI300X = GPUInfo.from_family(
    family=AMDCDNA3Family,
    name="MI300X",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx942",
    compute=9.4,
    version="CDNA3",
    sm_count=304,
)
"""AMD MI300X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# MI355X
# ===-----------------------------------------------------------------------===#


fn _get_mi355x_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD MI355X GPU.

    Returns:
        MLIR target configuration for MI355X.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx950", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime MI355X = GPUInfo.from_family(
    family=AMDCDNA4Family,
    name="MI355X",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx950",
    compute=9.5,
    version="CDNA4",
    sm_count=256,
)
"""AMD MI355X GPU configuration."""


# ===-----------------------------------------------------------------------===#
# Radeon 7xxx, 9xxx, 780m
# ===-----------------------------------------------------------------------===#


fn _get_9070_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 9070 GPU.

    Returns:
        MLIR target configuration for 9070.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1201", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_9060_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 9060 GPU.

    Returns:
        MLIR target configuration for 9060.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1200", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7900_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7900 GPU.

    Returns:
        MLIR target configuration for 7900.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1100", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7800_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7800/7700 GPU.

    Returns:
        MLIR target configuration for 7800/7700.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1101", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_7600_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 7600 GPU.

    Returns:
        MLIR target configuration for 7600.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1102", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_6900_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 6900 GPU.

    Returns:
        MLIR target configuration for 6900.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1030", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_780m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 780m GPU.

    Returns:
        MLIR target configuration for 780m.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1103", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_880m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 880M GPU.

    Returns:
        MLIR target configuration for 880M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1150", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_8060s_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 8060S GPU.

    Returns:
        MLIR target configuration for 8060S.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1151", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_860m_target() -> _TargetType:
    """Creates an MLIR target configuration for AMD Radeon 860M GPU.

    Returns:
        MLIR target configuration for 860M.
    """
    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx1152", `,
        `features = "", `,
        `data_layout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


comptime Radeon9070 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9070",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1201",
    compute=12.0,
    version="RDNA4",
    sm_count=64,
)
"""AMD Radeon 9070 GPU configuration."""

comptime Radeon9060 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 9060",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1200",
    compute=12.0,
    version="RDNA4",
    sm_count=32,
)
"""AMD Radeon 9060 GPU configuration."""

comptime Radeon7900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7900",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1100",
    compute=11.0,
    version="RDNA3",
    sm_count=96,
)
"""AMD Radeon 7900 GPU configuration."""

comptime Radeon7800 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7800/7700",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1101",
    compute=11.0,
    version="RDNA3",
    sm_count=60,
)
"""AMD Radeon 7800/7700 GPU configuration."""

comptime Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)
"""AMD Radeon 7600 GPU configuration."""

comptime Radeon6900 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 6900",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1102",
    compute=10.3,
    version="RDNA2",
    sm_count=60,
)
"""AMD Radeon 6900 GPU configuration."""


comptime Radeon780m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 780M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1103",
    compute=11.0,
    version="RDNA3",
    sm_count=12,
)
"""AMD Radeon 780M GPU configuration."""

comptime Radeon880m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 880M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1150",
    compute=11.5,
    version="RDNA3.5",
    sm_count=12,
)
"""AMD Radeon 880M GPU configuration."""

comptime Radeon8060s = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 8060S",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1151",
    compute=11.5,
    version="RDNA3.5",
    sm_count=40,
)
"""AMD Radeon 8060S GPU configuration."""

comptime Radeon860m = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 860M",
    vendor=Vendor.AMD_GPU,
    api="hip",
    arch_name="gfx1152",
    compute=11.5,
    version="RDNA3.5",
    sm_count=8,
)
"""AMD Radeon 860M GPU configuration."""


# ===-----------------------------------------------------------------------===#
# GPUInfo
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct GPUInfo(Equatable, RegisterPassable, Stringable, Writable):
    """Comprehensive information about a GPU architecture.

    This struct contains detailed specifications about GPU capabilities,
    including compute units, memory, thread organization, and performance
    characteristics.
    """

    var name: StaticString
    """The model name of the GPU."""

    var vendor: Vendor
    """The vendor/manufacturer of the GPU (e.g., NVIDIA, AMD)."""

    var api: StaticString
    """The graphics/compute API supported by the GPU (e.g., CUDA, ROCm)."""

    var arch_name: StaticString
    """The architecture name of the GPU (e.g., sm_80, gfx942)."""

    var compute: Float32
    """Compute capability version number for NVIDIA GPUs."""

    var version: StaticString
    """Version string of the GPU architecture."""

    var sm_count: Int
    """Number of streaming multiprocessors (SMs) on the GPU."""

    var warp_size: Int
    """Number of threads in a warp/wavefront."""

    var threads_per_multiprocessor: Int
    """Maximum number of threads per streaming multiprocessor."""

    var shared_memory_per_multiprocessor: Int
    """Size of shared memory available per multiprocessor in bytes."""

    var max_registers_per_block: Int
    """Maximum number of registers that can be allocated to a thread block."""

    var max_thread_block_size: Int
    """Maximum number of threads allowed in a thread block."""

    fn target(self) -> _TargetType:
        """Gets the MLIR target configuration for this GPU.

        Returns:
            MLIR target configuration for the GPU.
        """
        if self.name == "NVIDIA Tesla P100":
            return _get_teslap100_target()
        if self.name == "NVIDIA GeForce GTX 1060":
            return _get_gtx1060_target()
        if self.name == "NVIDIA GeForce GTX 1080 Ti":
            return _get_gtx1080ti_target()
        if self.name == "NVIDIA GeForce GTX 970":
            return _get_gtx970_target()
        if self.name == "RTX2060":
            return _get_rtx2060_target()
        if self.name == "NVIDIA GeForce RTX 3090":
            return _get_rtx3090_target()
        if self.name == "A100":
            return _get_a100_target()
        if self.name == "A10":
            return _get_a10_target()
        if self.name == "L4":
            return _get_l4_target()
        if self.name == "RTX4090m":
            return _get_rtx4090m_target()
        if self.name == "RTX4090":
            return _get_rtx4090_target()
        if self.name == "H100":
            return _get_h100_target()
        if self.name == "B100" or self.name == "B200":
            return _get_b100_target()
        if self.name == "DGX Spark":
            return _get_dgx_spark_target()
        if self.name == "RTX5090":
            return _get_rtx5090_target()
        if self.name == "Jetson Thor":
            return _get_jetson_thor_target()
        if self.name == "MI300X":
            return _get_mi300x_target()
        if self.name == "MI355X":
            return _get_mi355x_target()
        if self.name == "Radeon 780M":
            return _get_780m_target()
        if self.name == "Radeon 880M":
            return _get_880m_target()
        if self.name == "Radeon 8060S":
            return _get_8060s_target()
        if self.name == "Radeon 860M":
            return _get_860m_target()
        if self.name == "Radeon 6900":
            return _get_6900_target()
        if self.name == "Radeon 7900":
            return _get_7900_target()
        if self.name == "Radeon 7800/7700":
            return _get_7800_target()
        if self.name == "Radeon 7600":
            return _get_7600_target()
        if self.name == "Radeon 9070":
            return _get_9070_target()
        if self.name == "Radeon 9060":
            return _get_9060_target()
        if self.name == "M1":
            return _get_metal_m1_target()
        if self.name == "M2":
            return _get_metal_m2_target()
        if self.name == "M3":
            return _get_metal_m3_target()
        if self.name == "M4":
            return _get_metal_m4_target()
        if self.name == "M5":
            return _get_metal_m5_target()

        if self.name == "":
            return _get_empty_target()
        return _get_a100_target()

    @staticmethod
    fn from_target[target: _TargetType]() -> Self:
        """Creates a `GPUInfo` instance from an MLIR target.

        Parameters:
            target: MLIR target configuration.

        Returns:
            GPU info corresponding to the target.
        """
        return _get_info_from_target[CompilationTarget[target]._arch()]()

    @staticmethod
    fn from_name[name: StaticString]() -> Self:
        """Creates a `GPUInfo` instance from a GPU architecture name.

        Parameters:
            name: GPU architecture name (e.g., "sm_80", "gfx942").

        Returns:
            GPU info corresponding to the architecture name.
        """
        return _get_info_from_target[name]()

    @staticmethod
    fn from_family(
        family: AcceleratorArchitectureFamily,
        name: StaticString,
        vendor: Vendor,
        api: StaticString,
        arch_name: StaticString,
        compute: Float32,
        version: StaticString,
        sm_count: Int,
    ) -> Self:
        """Creates a `GPUInfo` instance using architecture family defaults.

        This constructor simplifies GPU definition by inheriting common
        characteristics from an architecture family while allowing specific
        values to be overridden.

        Args:
            family: Architecture family providing default values.
            name: The model name of the GPU.
            vendor: The vendor/manufacturer of the GPU.
            api: The graphics/compute API supported by the GPU.
            arch_name: The architecture name of the GPU.
            compute: Compute capability version number.
            version: Version string of the GPU architecture.
            sm_count: Number of streaming multiprocessors.

        Returns:
            A fully configured GPUInfo instance.
        """
        return Self(
            name=name,
            vendor=vendor,
            api=api,
            arch_name=arch_name,
            compute=compute,
            version=version,
            sm_count=sm_count,
            warp_size=family.warp_size,
            threads_per_multiprocessor=family.threads_per_multiprocessor,
            shared_memory_per_multiprocessor=family.shared_memory_per_multiprocessor,
            max_registers_per_block=family.max_registers_per_block,
            max_thread_block_size=family.max_thread_block_size,
        )

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `GPUInfo` instances represent the same GPU model.

        Args:
            other: Another `GPUInfo` instance to compare against.

        Returns:
            True if both instances represent the same GPU model.
        """
        return self.name == other.name

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes GPU information to a writer.

        Outputs all GPU specifications and capabilities to the provided writer
        in a human-readable format.

        Args:
            writer: A Writer instance to output the GPU information.
        """
        writer.write("name: ", self.name, "\n")
        writer.write("vendor: ", self.vendor, "\n")
        writer.write("api: ", self.api, "\n")
        writer.write("arch_name: ", self.arch_name, "\n")
        writer.write("compute: ", self.compute, "\n")
        writer.write("version: ", self.version, "\n")
        writer.write("sm_count: ", self.sm_count, "\n")
        writer.write("warp_size: ", self.warp_size, "\n")
        writer.write(
            "threads_per_multiprocessor: ",
            self.threads_per_multiprocessor,
            "\n",
        )
        writer.write(
            "shared_memory_per_multiprocessor: ",
            self.shared_memory_per_multiprocessor,
            "\n",
        )
        writer.write(
            "max_registers_per_block: ", self.max_registers_per_block, "\n"
        )
        writer.write(
            "max_thread_block_size: ", self.max_thread_block_size, "\n"
        )

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the GPU information.

        Converts all GPU specifications and capabilities to a human-readable
        string format.

        Returns:
            String containing all GPU information.
        """
        return String.write(self)


# ===-----------------------------------------------------------------------===#
# _build_unsupported_arch_error
# ===-----------------------------------------------------------------------===#


fn _build_unsupported_arch_error[target_arch: StaticString]() -> String:
    """Builds a helpful error message for unsupported GPU architectures.

    Provides a comprehensive list of all supported GPU architectures across
    all vendors with documentation links.

    Parameters:
        target_arch: The unsupported target architecture string.

    Returns:
        A detailed error message with supported architectures and doc links.
    """
    comptime nvidia_archs = "sm_52 (Maxwell), sm_60/sm_61 (Pascal), sm_75 (Turing), sm_80 (Ampere A100), sm_86 (Ampere A10), sm_87 (Orin), sm_89 (Ada L4/RTX4090), sm_90/sm_90a (Hopper H100), sm_100/sm_100a (Blackwell B100/B200), sm_110 (Jetson Thor), sm_120/sm_120a (Blackwell RTX5090), sm_121 (DGX Spark)"
    comptime amd_archs = "gfx942 (MI300X), gfx950 (MI355X), gfx1030 (Radeon 6900), gfx1100 (Radeon 7900), gfx1101 (Radeon 7800), gfx1102 (Radeon 7600), gfx1103 (Radeon 780M), gfx1150/gfx1151/gfx1152 (Radeon 8xx), gfx1200 (Radeon 9060), gfx1201 (Radeon 9070)"
    comptime apple_archs = "metal:1 (M1), metal:2 (M2), metal:3 (M3), metal:4 (M4)"

    var prefix: String

    comptime if target_arch == "":
        prefix = "Unknown GPU architecture detected."
    else:
        prefix = String(
            "GPU architecture '", target_arch, "' is not supported."
        )

    return String(
        prefix,
        "\n\nSupported GPU architectures:\n\n",
        "  NVIDIA: ",
        nvidia_archs,
        "\n  See: https://developer.nvidia.com/cuda-gpus\n\n",
        "  AMD: ",
        amd_archs,
        (
            "\n  See:"
            " https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html"
            "\n\n"
        ),
        "  Apple: ",
        apple_archs,
        (
            "\n  See:"
            " https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"
        ),
    )


# ===-----------------------------------------------------------------------===#
# _get_info_from_target
# ===-----------------------------------------------------------------------===#

# All supported target architectures in canonical form.
# This is the canonical list used for validation in _get_info_from_target.
# Normalization: "nvidia:80" -> "sm_80", "mi300x" -> "gfx942",
#                "amdgpu:gfx942" -> "gfx942", "metal:4" -> "apple-m4".
#
# SYNC: This list must stay in sync with printSupportedAccelerators() in
#       KGEN/tools/mojo/Build/mojo-build.cpp. Run the following test to verify:
#       bazel test //KGEN/test/mojo-tool:build/verify_supported_accelerators_sync.mojo.test
comptime _all_targets = (
    StaticString("sm_52"),
    StaticString("sm_60"),
    StaticString("sm_61"),
    StaticString("sm_75"),
    StaticString("sm_80"),
    StaticString("sm_86"),
    StaticString("sm_87"),
    StaticString("sm_89"),
    StaticString("sm_90"),
    StaticString("sm_90a"),
    StaticString("sm_100"),
    StaticString("sm_100a"),
    StaticString("sm_110"),
    StaticString("sm_110a"),
    StaticString("sm_120"),
    StaticString("sm_120a"),
    StaticString("sm_121"),
    StaticString("sm_121a"),
    StaticString("gfx942"),
    StaticString("gfx950"),
    StaticString("gfx1030"),
    StaticString("gfx1100"),
    StaticString("gfx1101"),
    StaticString("gfx1102"),
    StaticString("gfx1103"),
    StaticString("gfx1150"),
    StaticString("gfx1151"),
    StaticString("gfx1152"),
    StaticString("gfx1200"),
    StaticString("gfx1201"),
    StaticString("apple-m1"),
    StaticString("apple-m2"),
    StaticString("apple-m3"),
    StaticString("apple-m4"),
    StaticString("apple-m5"),
    StaticString("cuda"),
)


@always_inline
fn _get_info_from_target[target_arch0: StaticString]() -> GPUInfo:
    """Gets `GPUInfo` for a specific target architecture.

    Maps target architecture strings to corresponding `GPUInfo` instances.

    Parameters:
        target_arch0: Target architecture string (e.g., "sm_80", "gfx942").

    Returns:
        `GPUInfo` instance for the specified target architecture.
    """
    # Normalize the target architecture to canonical form.
    # NVIDIA: "nvidia:sm_90a" -> "sm_90a", "nvidia:sm90" -> "sm_90", "nvidia:80" -> "sm_80", "sm80" -> "sm_80"
    # AMD: "mi300x" -> "gfx942", "mi355x" -> "gfx950", "amdgpu:gfx942" -> "gfx942"
    # Apple: "metal:4" -> "apple-m4"
    comptime target_arch = (
        target_arch0
        # NVIDIA normalization
        .replace("nvidia:sm_", "sm_")
        .replace("nvidia:sm", "sm_")
        .replace("nvidia:", "sm_")
        .replace("sm", "sm_")
        .replace("sm__", "sm_")
        # AMD normalization
        .replace("mi300x", "gfx942")
        .replace("mi355x", "gfx950")
        .replace("amdgpu:", "")
        # Apple normalization
        .replace("metal:", "apple-m")
    )

    comptime assert (
        StaticString(target_arch) in _all_targets
    ), _build_unsupported_arch_error[target_arch0]()

    comptime if target_arch == "sm_52":
        return materialize[GTX970]()
    elif target_arch == "sm_60":
        return materialize[TeslaP100]()
    elif target_arch == "sm_61":
        # FIXME GTX1060 and GTX1080Ti architecture wise are different (sm_count is different). We need to differentiate between them here at compile time.
        # return materialize[GTX1060]()
        return materialize[GTX1080Ti]()
    elif target_arch == "sm_75":
        return materialize[RTX2060]()
    elif target_arch == "sm_80":
        return materialize[A100]()
    elif target_arch == "sm_86":
        return materialize[A10]()
    elif target_arch == "sm_87":
        return materialize[OrinNano]()
    elif target_arch == "sm_89":
        return materialize[L4]()
    elif target_arch == "sm_90" or target_arch == "sm_90a":
        return materialize[H100]()
    elif target_arch == "sm_100" or target_arch == "sm_100a":
        # FIXME (KERN-1814): Unlike H100 and H200, blackwell devices (B100 vs B200)
        # architecture wise are different. We need to differentiate between them here.
        return materialize[B200]()
    elif target_arch == "sm_110" or target_arch == "sm_110a":
        return materialize[JetsonThor]()
    elif target_arch == "sm_120" or target_arch == "sm_120a":
        return materialize[RTX5090]()
    elif target_arch == "sm_121" or target_arch == "sm_121a":
        return materialize[DGXSpark]()
    # AMD (gfx IDs; "mi300x"/"mi355x" aliases are normalized to gfx IDs above)
    elif target_arch == "gfx942":
        return materialize[MI300X]()
    elif target_arch == "gfx950":
        return materialize[MI355X]()
    elif target_arch == "gfx1030":
        return materialize[Radeon6900]()
    elif target_arch == "gfx1100":
        return materialize[Radeon7900]()
    elif target_arch == "gfx1101":
        return materialize[Radeon7800]()
    elif target_arch == "gfx1102":
        return materialize[Radeon7600]()
    elif target_arch == "gfx1103":
        return materialize[Radeon780m]()
    elif target_arch == "gfx1150":
        return materialize[Radeon880m]()
    elif target_arch == "gfx1151":
        return materialize[Radeon8060s]()
    elif target_arch == "gfx1152":
        return materialize[Radeon860m]()
    elif target_arch == "gfx1200":
        return materialize[Radeon9060]()
    elif target_arch == "gfx1201":
        return materialize[Radeon9070]()
    elif target_arch == "apple-m1":
        return materialize[MetalM1]()
    elif target_arch == "apple-m2":
        return materialize[MetalM2]()
    elif target_arch == "apple-m3":
        return materialize[MetalM3]()
    elif target_arch == "apple-m4":
        return materialize[MetalM4]()
    elif target_arch == "apple-m5":
        return materialize[MetalM5]()
    # "cuda" means generic CUDA — use runtime GPU detection.
    elif target_arch == "cuda":
        return _get_info_from_target[_accelerator_arch()]()
    elif _accelerator_arch() == "":
        return materialize[NoGPU]()
    else:
        return _get_info_from_target[_accelerator_arch()]()


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn is_gpu[target: StringSlice]() -> Bool:
    """Checks if the target is a GPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return is_gpu(target)


fn is_gpu(target: StringSlice) -> Bool:
    """Checks if the target is a GPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a GPU, False otherwise.
    """
    return target == "gpu"


fn is_cpu[target: StringSlice]() -> Bool:
    """Checks if the target is a CPU (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return is_cpu(target)


fn is_cpu(target: StringSlice) -> Bool:
    """Checks if the target is a CPU (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is a CPU, False otherwise.
    """
    return target == "cpu"


fn is_valid_target[target: StringSlice]() -> Bool:
    """Checks if the target is valid (compile-time version).

    Parameters:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_valid_target(target)


fn is_valid_target(target: StringSlice) -> Bool:
    """Checks if the target is valid (runtime version).

    Args:
        target: Target string to check.

    Returns:
        True if the target is valid (CPU or GPU), False otherwise.
    """
    return is_gpu(target) or is_cpu(target)
