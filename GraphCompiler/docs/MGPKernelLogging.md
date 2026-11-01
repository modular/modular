# MGP Kernel Logging

## Overview

The kernel logging feature provides visibility into MGP kernel execution by
logging kernel launch and completion events. This is an **internal development
feature** for Modular engineers to debug performance issues, understand
execution flow, and trace kernel behavior during model inference.

When enabled at compile time, the system outputs timestamped log messages for
all kernel operations, including device information (CPU vs GPU) and precise
timing data.

**Note:** This feature is disabled by default and requires recompilation to
enable.

## Quick Start

In the `GraphCompiler/lib/MGPDialect/Primitives/MGPPrimitives.cpp`
file, enable the define `ENABLE_MGP_KERNEL_LOGGING`.

Tests for this logging capability automatically enable that define for this
build target:

```bash
./bazelw test //GraphCompiler/test/unittests:KernelLoggerTests
```

Example output when enabled:

```bash
[KERNEL] SYNC LAUNCH matmul_kernel on GPU:0 [id=1] at +1.234ms
[KERNEL] SYNC COMPLETE matmul_kernel on GPU:0 [id=1] at +2.456ms
[KERNEL] ASYNC LAUNCH custom__vroom_q6_k_repack_weights on CPU [id=2] at +2.789ms
[KERNEL] ASYNC ERROR custom__vroom_q6_k_repack_weights on CPU [id=2] at +3.012ms
[KERNEL] SYNC LAUNCH cpu_normalization on CPU [id=3] at +3.100ms
[KERNEL] SYNC COMPLETE cpu_normalization on CPU [id=3] at +3.125ms
```

## Configuration

### Compile-Time Configuration

Kernel logging is controlled at compile time via the `ENABLE_MGP_KERNEL_LOGGING`
preprocessor define:

```cpp
// Enabled in GraphCompiler/lib/MGPDialect/Primitives/MGPPrimitives.cpp
#if ENABLE_MGP_KERNEL_LOGGING
  // Logging code paths active
#else
  // No-op logging (default)
#endif
```

### Build Targets

- **Standard builds**: `//GraphCompiler:MGPPrimitives` - No logging (default)
- **Logging builds**: `//GraphCompiler:MGPPrimitivesWithLogging` - Always logs
- **Test target**: `//GraphCompiler/test/unittests:KernelLoggerTests` - Uses
  logging-enabled library

## Log Format

Each log entry follows this format:

```bash
[KERNEL] <EXECUTION_TYPE> <ACTION> <kernel_name> on <device_info> [id=<kernel_id>] at +<timestamp>ms
```

### Components

- **EXECUTION_TYPE**: Either `SYNC` (synchronous) or `ASYNC` (asynchronous)
- **ACTION**:
  - `LAUNCH` - Kernel started
  - `COMPLETE` - Kernel finished successfully
  - `ERROR` - Kernel failed
- **kernel_name**: Name/identifier of the executed kernel
- **device_info**: Device where kernel executed:
  - `CPU`: CPU execution
  - `GPU:N`: GPU execution on device N (e.g., `GPU:0`, `GPU:1`)
  - `Multi(device1,device2,...)`: Multi-device execution (e.g.,
    `Multi(GPU:0,GPU:1,CPU)`)
- **kernel_id**: Unique identifier to correlate launch and completion events
- **timestamp**: Relative timestamp in milliseconds since logging started, with
  microsecond precision
