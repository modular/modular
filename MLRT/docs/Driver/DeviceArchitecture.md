# AsyncRT Device Architecture

This document describes the architecture of the AsyncRT device layer,
including the relationships between `Device`, `DeviceContext`,
`CompilationDevice`, and driver components.

## Overview

The AsyncRT device architecture provides a unified abstraction over
heterogeneous compute devices (CUDA GPUs, AMD GPUs, Apple GPUs, and CPUs) with
two primary execution paths:

1. **Runtime Path**: Execute operations on physical or virtual hardware
2. **Compilation Path**: Compile device code (e.g., PTX → CUBIN) without
   requiring physical hardware

## Architecture Layers

The device stack consists of several layers, from lowest to highest:

```text
┌─────────────────────────────────────────────────────────────┐
│                    User Code / SDK                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DeviceContext                            │
│  • Stream-based execution context (1:1 with stream)         │
│  • Buffer management, kernel execution                      │
│  • Delegates compilation to Device                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       Device                                │
│  • Singleton per physical/virtual device                    │
│  • Memory management, context creation                      │
│  • Owns CompilationDevice                                   │
│  • Owns or references Driver                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴────────────────────┐
        ↓                                          ↓
┌──────────────────┐                    ┌──────────────────┐
│  Driver          │                    │ CompilationDevice│
│  (Runtime)       │                    │  (Compilation)   │
├──────────────────┤                    ├──────────────────┤
│ • CUDADriver     │                    │ • CUDA:          │
│ • HIPDriver      │                    │   CUDACompilation│
│ • MetalDriver    │                    │   Device         │
│                  │                    │ • HIP:           │
│ Wraps:           │                    │   HIPCompilation │
│ • libcuda.so.1   │                    │   Device         │
│ • libamdhip64.so │                    │ • Others: nullptr│
│ • Metal.framework│                    │                  │
└──────────────────┘                    │ CUDA wraps:      │
                                        │ • libnvptx       │
                                        │   compiler.so    │
                                        │                  │
                                        │ HIP stores arch: │
                                        │ • gfx942, gfx950 │
                                        │ • (KGEN handles  │
                                        │   compilation)   │
                                        └──────────────────┘
                                                  ↓
                                        ┌──────────────────┐
                                        │ CompilationDriver│
                                        ├──────────────────┤
                                        │ • CUDACompilation│
                                        │   Driver         │
                                        │   (CUDA only)    │
                                        │                  │
                                        │ Wraps:           │
                                        │ • nvPTXCompiler  │
                                        │   API            │
                                        └──────────────────┘
```

## Component Responsibilities

### DeviceContext

**Purpose**: Represents a stream of execution on a device.

**Responsibilities**:

- Enqueue operations (kernel execution, memory copies, etc.) to a stream
- Manage buffer allocation through the device's memory manager
- Synchronization between streams via events
- **Delegates** all compilation operations to its parent `Device`

**Key Properties**:

- Multiple `DeviceContext` objects per `Device` (N:1 relationship)
- Each `DeviceContext` maps to exactly one execution stream
- No longer owns or knows about `CompilationDevice`

**Example Usage**:

```cpp
auto deviceOrErr = CUDADevice::get(0);  // Get GPU 0
auto ctxOrErr = device.createContext(); // Create execution context
ctx.compileFunction(...);               // Delegates to device
```

### Device

**Purpose**: Singleton representing a physical or virtual compute device.

**Responsibilities**:

- Device-level resource management (memory info, capabilities)
- Create execution contexts (`DeviceContext`)
- **Own** compilation device (if compilation is supported)
- Coordinate between runtime and compilation operations
- Cache compiled functions

**Key Properties**:

- One `Device` instance per physical/virtual device
- Owns `CompilationDevice` (unique_ptr, may be null)
- References `Driver` for runtime operations (const reference)
- Provides unified API across all device types

**Compilation Support by Device Type**:

```cpp
// CUDA Device - has compilation
class CUDADevice : public Device {
  std::unique_ptr<CUDACompilationDevice> compilationDevice_;
  const CUDACompilationDevice* compilationDevice() const override {
    return compilationDevice_.get();
  }
};

// Virtual Device - has compilation
class VirtualDevice : public Device {
  std::unique_ptr<CompilationDevice> compilationDevice_;
  const CompilationDevice* compilationDevice() const override {
    return compilationDevice_.get();
  }
};

// HIP Device - has compilation device (architecture storage only)
class HIPDevice : public Device {
  std::unique_ptr<HIPCompilationDevice> compilationDevice_;
  const CompilationDevice* compilationDevice() const override {
    return compilationDevice_.get();
  }
};

// CPU/Metal - no compilation yet
class CpuDevice : public Device {
  const CompilationDevice* compilationDevice() const override {
    return nullptr;
  }
};
```

### CompilationDevice

**Purpose**: Abstract interface for device code compilation without requiring
physical hardware.

**Responsibilities**:

- Compile intermediate representations (e.g., PTX) to device code (e.g., CUBIN)
- Provide architecture-specific compilation (sm_80, sm_90, etc.)
- Independent of runtime GPU contexts
- Uses `CompilationDriver` for actual compilation

**Key Properties**:

- Owned by `Device` (not `DeviceContext`)
- Optional: returns `nullptr` for devices without compilation support
- Stateless: each compilation is independent

**Current Implementations**:

- `CUDACompilationDevice`: Compiles PTX → CUBIN using nvPTXCompiler
- `HIPCompilationDevice`: Stores target architecture for AMD GPUs
  (gfx942, gfx950, etc.)
  - **Note**: Unlike CUDA, HIP does not perform runtime compilation
  - All AMD GPU compilation (LLVM IR → object files → shared objects) is
    handled by KGEN through the LLVM AMDGPU backend and ld.lld linker
  - The CompilationDevice exists primarily to store and provide the target
    architecture
- Future: `MetalCompilationDevice`

### Driver (Runtime Driver)

**Purpose**: Thin wrapper around vendor-specific runtime libraries (libcuda,
libamdhip64, etc.).

**Responsibilities**:

- Load and manage vendor library symbols
- Provide type-safe C++ wrappers around C APIs
- Handle runtime operations: kernel launch, memory management,
  synchronization
- Singleton per driver type

**Examples**:

- `CUDADriver`: Wraps CUDA Driver API (libcuda.so.1)
- `HIPDriver`: Wraps HIP Runtime (libamdhip64.so)
- Metal: Uses Metal.framework directly

**Key Properties**:

- Loaded once per process (singleton)
- Used by `Device` for all runtime operations
- Separate from `CompilationDriver`

### CompilationDriver

**Purpose**: Thin wrapper around vendor-specific compilation libraries.

**Responsibilities**:

- Provide type-safe wrappers around compilation APIs
- Handle code generation operations only (no runtime operations)
- Singleton per driver type

**Examples**:

- `CUDACompilationDriver`: Wraps nvPTXCompiler (libnvptxcompiler.so)

**Key Properties**:

- Loaded independently of runtime driver
- Used by `CompilationDevice` for code generation
- Can work without GPU hardware present

## Ownership Model

The ownership hierarchy follows these rules:

```text
Process (singleton)
├── Driver (singleton, shared across devices)
│   └── Example: CUDADriver (libcuda.so.1)
│
├── CompilationDriver (singleton, shared across compilation devices)
│   └── Example: CUDACompilationDriver (libnvptxcompiler.so)
│
└── Device (one per physical/virtual device)
    ├── OWNS: CompilationDevice (unique_ptr, optional)
    │   └── REFERENCES: CompilationDriver (const reference)
    │
    ├── REFERENCES: Driver (const reference)
    │
    └── CREATES: DeviceContext (returned as RCRef)
        └── REFERENCES: Device (via RCRef/DeviceRef)
```

**Ownership Rules**:

1. **Driver Lifetime**: Global singletons, live for process lifetime
2. **Device Lifetime**: Managed by static factory methods, typically process
   lifetime
3. **CompilationDevice Lifetime**: Owned by `Device`, lives as long as device
4. **DeviceContext Lifetime**: Reference-counted, can outlive user code but
   references device

## Execution Paths

### Runtime Path

Used for executing operations on hardware:

```text
User Code
    ↓
DeviceContext::enqueueFunctionExecDirect()
    ↓
DeviceContext::loadFunction()  // JIT compile if needed
    ↓
CUDADeviceFunction
    ↓
CUDADriver::cuLaunchKernel()
    ↓
CUDA Driver API (libcuda.so.1)
```

### Compilation Path

Used for compiling device code (does not require hardware):

```text
User Code / DeviceContext::loadFunction()
    ↓
DeviceContext::compileFunction()
    ↓
Device::compileFunction()
    ↓
Device::compilationDevice()  // Returns CompilationDevice* or nullptr
    ↓
CompilationDevice::compileFunction()  // e.g., CUDACompilationDevice
    ↓
CUDACompilationDriver::NVPTXCompilerCompile()
    ↓
nvPTXCompiler API (libnvptxcompiler.so)
```

**Key Points**:

- Compilation path is **independent** of runtime path
- `CompilationDevice` does NOT require GPU hardware or driver context
- Enables "compile-only" mode for model compilation without GPUs

**HIP Compilation Path (AMD GPUs)**:

HIP does not use the `compileFunction()` path shown above. Instead:

```text
KGEN ObjectCompiler
    ↓
LLVM LLC (AMDGPU backend) - compile LLVM IR → Object file (.o)
    ↓
ld.lld linker - link object file → Shared object (.so with HSACO)
    ↓
Load .so at runtime
```

The `HIPCompilationDevice` exists to:

1. Store the target architecture string (e.g., "gfx950" for MI355X)
2. Provide architecture via `getArchName()` to KGEN's ObjectCompiler
3. Enable virtual device creation for compile-only mode

**Key difference**: CUDA requires runtime compilation (PTX→CUBIN), while HIP
compiles everything through LLVM at build time.

## Device Type Details

### CUDA Device

**Components**:

```cpp
CUDADevice
├── OWNS: CUDACompilationDevice
│   └── USES: CUDACompilationDriver (nvPTXCompiler)
└── REFERENCES: CUDADriver (CUDA Driver API)
```

**Capabilities**:

- ✅ Runtime execution (via CUDADriver)
- ✅ Compilation (PTX → CUBIN via CUDACompilationDevice)
- ✅ Compile-only mode (CUDACompilationDevice works without GPU)

**Architecture Flow**:

```text
CUDADeviceContext → CUDADevice → CUDADriver (runtime)
                        ↓
                    CUDACompilationDevice → CUDACompilationDriver (compile)
```

### Virtual Device

**Components**:

```cpp
VirtualDevice
├── OWNS: CompilationDevice (can be CUDA, HIP, or Metal)
│   ├── CUDACompilationDevice → CUDACompilationDriver (nvPTXCompiler)
│   ├── HIPCompilationDevice → No driver (KGEN handles compilation)
│   └── MetalCompilationDevice (future)
└── NO RUNTIME DRIVER (virtual device)
```

**Purpose**: Enable compile-only workflows without physical hardware.

**Capabilities**:

- ❌ Runtime execution (no physical hardware)
- ✅ Compilation (delegates to appropriate CompilationDevice)
- ✅ Compile-only mode (primary use case)

**Supported APIs**:

- **CUDA**: Uses `CUDACompilationDevice` for PTX → CUBIN compilation
- **HIP**: Uses `HIPCompilationDevice` to store target architecture (gfx942,
  gfx950)
  - Compilation happens through KGEN's LLVM AMDGPU backend
  - No runtime compilation needed
- **Metal**: Future support

**Use Cases**:

- CI/CD pipelines without GPUs
- Model compilation services
- Cross-compilation for target architectures

**Example - Virtual HIP Device**:

```cpp
// Create virtual AMD GPU device for gfx950 (MI355X)
auto compilationDevOrErr = HIPCompilationDevice::create("gfx950");
auto virtualDevice = new VirtualDevice(
    0, "virtual-gfx950", std::move(*compilationDevOrErr)
);

// Architecture is available for KGEN
auto archOrErr = virtualDevice->getArchName();  // Returns "gfx950"
// ✅ KGEN uses this to compile for AMD MI355X without physical GPU
```

### CPU Device

**Components**:

```cpp
CpuDevice
└── NO COMPILATION DEVICE
└── NO DRIVER (uses standard CPU operations)
```

**Capabilities**:

- ✅ Runtime execution (native CPU code)
- ❌ Compilation (not yet implemented)

### HIP Device

**Components**:

```cpp
HIPDevice
├── OWNS: HIPCompilationDevice (stores target architecture)
│   └── NO CompilationDriver (compilation handled by KGEN)
└── REFERENCES: HIPDriver (AMD HIP Runtime)
```

**Capabilities**:

- ✅ Runtime execution (via HIPDriver)
- ✅ Compilation (via KGEN's LLVM AMDGPU backend)
- ✅ Compile-only mode (HIPCompilationDevice works without GPU)

**Architecture Flow**:

Unlike CUDA, AMD GPU compilation does not use runtime compilation:

```text
NVIDIA (CUDA):
  LLVM IR → PTX (via LLC) → CUBIN (via nvPTXCompiler runtime compilation)
                                    ↑
                            CUDACompilationDevice performs this step

AMD (HIP):
  LLVM IR → Object file (.o) → Shared object (.so)
      ↑              ↑                  ↑
   KGEN's LLC   LLVM AMDGPU      ld.lld linker
                  backend

  HIPCompilationDevice only stores target arch (gfx942, gfx950, etc.)
  - No runtime compilation needed
  - All compilation happens at build time through LLVM
```

**Key Differences from CUDA**:

1. **No Runtime Compilation**: HIP does not require a compilation driver like
   nvPTXCompiler
2. **LLVM Backend**: All AMD GPU code generation uses LLVM's AMDGPU backend
   directly
3. **Shared Objects**: HIP produces `.so` files containing HSACO (HSA Code
   Object) format
4. **Architecture Storage**: `HIPCompilationDevice` exists primarily to store
   and provide the target architecture string to KGEN

### Metal Device

**Components**:

```cpp
MetalDevice
├── NO COMPILATION DEVICE (future: MetalCompilationDevice)
└── USES: Metal.framework directly
```

**Capabilities**:

- ✅ Runtime execution (via Metal framework)
- ❌ Compilation (future work)

## Example: CUDA Compilation Flow

Let's trace a complete compilation from user code to compiled CUBIN:

```cpp
// 1. Get CUDA device (loads CUDADriver and creates CUDACompilationDevice)
auto deviceOrErr = CUDADevice::get(0);
const CUDADevice& device = *deviceOrErr;

// 2. Create execution context
auto ctxOrErr = device.createContext();
const CUDADeviceContext& ctx = *ctxOrErr;

// 3. User requests compilation
auto cubinOrErr = ctx.compileFunction(ptxCode, "debug", 3, "sm_80");
```

**Internal Flow**:

```text
DeviceContext::compileFunction(ptxCode, ...)
    ↓
Device::compileFunction(ptxCode, ...)
    ↓
device.compilationDevice()  // Returns CUDACompilationDevice*
    ↓
CUDACompilationDevice::compileFunction(ptxCode, ...)
    ↓
CUDACompilationDriver::get()  // Get singleton driver
    ↓
driver.NVPTXCompilerCompile(...)
    ↓
::nvPTXCompilerCompile(...)  // Actual nvPTXCompiler API
    ↓
Returns compiled CUBIN buffer
```

## Example: Virtual Device for Compile-Only Mode

```cpp
// 1. Create virtual device with target architecture
auto compilationDevOrErr = CUDACompilationDevice::create("sm_90");
auto virtualDevice = new VirtualDevice(
    0, "virtual-sm_90", std::move(*compilationDevOrErr)
);

// 2. Create virtual context
auto ctxOrErr = virtualDevice->createContext();

// 3. Compile without GPU
auto cubinOrErr = ctx.compileFunction(ptxCode, "debug", 3, "sm_90");
// ✅ Works without any GPU hardware!
```

## Design Principles

### Separation of Concerns

1. **Runtime** (Driver) and **Compilation** (CompilationDriver) are completely
   separate
   - Different libraries: libcuda.so.1 vs libnvptxcompiler.so
   - Different initialization and error handling
   - Can load compilation driver without runtime driver

2. **Device** owns compilation capability, not **DeviceContext**
   - Compilation is device-level capability
   - Multiple contexts can share compilation device
   - Cleaner ownership model

### Compile-Only Mode

The architecture explicitly supports compilation without hardware:

```cpp
// Traditional: Requires GPU
CUDADevice → requires libcuda.so.1 → requires NVIDIA GPU

// Compile-Only: No GPU needed
VirtualDevice → CUDACompilationDevice → CUDACompilationDriver
    ↓                                           ↓
    No runtime driver                    libnvptxcompiler.so (CPU-only)
```

This enables:

- Building models in CI without GPU runners
- Cross-compilation for different GPU architectures
- Faster model deployment (pre-compile, then deploy)

### Extensibility

Adding a new device type requires:

1. Implement `Device` subclass (e.g., `VulkanDevice`)
2. Implement `Driver` wrapper (e.g., `VulkanDriver`)
3. Optionally implement `CompilationDevice` (e.g., `VulkanCompilationDevice`)
4. Optionally implement `CompilationDriver` (e.g., `VulkanCompilationDriver`)

Each component is independent and can be developed/tested separately.

## References

- `MLRT/include/MLRT/Driver/DeviceContext/DeviceContext.h`: Core device
  interfaces
- `MLRT/include/MLRT/Driver/DeviceContext/CompilationDevice.h`: Compilation
  interface
- `MLRT/lib/Driver/DeviceContext/CUDA/CUDADevice*.{h,cpp}`: CUDA implementation
- `MLRT/lib/Driver/DeviceContext/Virtual/VirtualDevice*.{h,cpp}`: Virtual device
  implementation
- `MLRT/docs/Driver/Driver.md`: Runtime driver overview
