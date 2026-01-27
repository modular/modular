# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test --print-* target information options.

# Test --print-supported-targets lists registered targets
# RUN: %mojo-build --print-supported-targets 2>&1 | FileCheck %s --check-prefix=CHECK_TARGETS

# Test --print-effective-target shows host target by default
# RUN: %mojo-build --print-effective-target 2>&1 | FileCheck %s --check-prefix=CHECK_EFFECTIVE

# Test --print-effective-target with cross-compilation options
# RUN: %mojo-build --print-effective-target --target-triple x86_64-unknown-linux-gnu --mcpu=haswell 2>&1 | FileCheck %s --check-prefix=CHECK_EFFECTIVE_CROSS

# Test --print-supported-cpus requires --target-triple
# RUN: not %mojo-build --print-supported-cpus 2>&1 | FileCheck %s --check-prefix=CHECK_CPUS_NO_TARGET

# Test --print-supported-cpus lists CPUs for specific target
# RUN: %mojo-build --print-supported-cpus --target-triple x86_64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=CHECK_CPUS_X86

# Test error when multiple print options specified
# RUN: not %mojo-build --print-effective-target --print-supported-targets 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_MULTI

# Test error for unsupported target
# RUN: not %mojo-build --print-supported-cpus --target-triple invalid-unknown-unknown 2>&1 | FileCheck %s --check-prefix=CHECK_INVALID_TARGET

# CHECK_TARGETS: Registered Targets:
# CHECK_TARGETS-DAG: aarch64
# CHECK_TARGETS-DAG: x86-64

# CHECK_EFFECTIVE: Effective target configuration:
# CHECK_EFFECTIVE: --target-triple
# CHECK_EFFECTIVE: --target-cpu
# CHECK_EFFECTIVE: --target-features

# CHECK_EFFECTIVE_CROSS: Effective target configuration:
# CHECK_EFFECTIVE_CROSS: --target-triple x86_64-unknown-linux-gnu
# CHECK_EFFECTIVE_CROSS: --target-cpu haswell
# CHECK_EFFECTIVE_CROSS: --target-features +avx

# CHECK_CPUS_NO_TARGET: error: --print-supported-cpus requires --target-triple to be specified

# CHECK_CPUS_X86: Available CPUs for target x86_64-unknown-linux-gnu:
# CHECK_CPUS_X86: haswell
# CHECK_CPUS_X86: skylake

# CHECK_ERROR_MULTI: error: only one --print-* option can be specified at a time

# CHECK_INVALID_TARGET: error: unknown target triple 'invalid-unknown-unknown'
# CHECK_INVALID_TARGET: Use --print-supported-targets to see available architectures.


fn main():
    pass
