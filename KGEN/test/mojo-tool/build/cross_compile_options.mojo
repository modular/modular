# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test cross-compilation target options and validation of option family mixing.
#
# There are two option families that should not be mixed:
# - LLVM-style: --target-cpu, --target-features
# - GCC/Clang-style: --march, --mcpu, --mtune

# Cross-compile to x86_64 Linux using --mcpu (should derive target CPU from mcpu)
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --mcpu=haswell --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_X86

# Cross-compile to AArch64 Linux using --mcpu
# RUN: %mojo-build --target-triple aarch64-unknown-linux-gnu --mcpu=cortex-a72 --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_AARCH64

# When both --march and --mcpu are specified, --march takes precedence for the arch name
# RUN: %mojo-build --target-triple x86_64-unknown-linux-gnu --march=x86-64 --mcpu=haswell --emit=llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK_MARCH

# Error when --target-cpu is used with --mcpu (mixing option families)
# RUN: not %mojo-build --target-cpu=haswell --mcpu=skylake %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_CPU_MCPU

# Error when --target-cpu is used with --march (mixing option families)
# RUN: not %mojo-build --target-cpu=haswell --march=x86-64 %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_CPU_MARCH

# Error when --target-features is used with --mcpu (mixing option families)
# RUN: not %mojo-build --mcpu=haswell --target-features="+avx512f" %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_FEATURES_MCPU

# Error when --target-features is used with --march (mixing option families)
# RUN: not %mojo-build --march=x86-64 --target-features="+avx2" %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_FEATURES_MARCH

# CHECK_ERROR_CPU_MCPU: error: --target-cpu cannot be used with --march or --mcpu
# CHECK_ERROR_CPU_MARCH: error: --target-cpu cannot be used with --march or --mcpu
# CHECK_ERROR_FEATURES_MCPU: error: --target-features cannot be used with --march or --mcpu
# CHECK_ERROR_FEATURES_MARCH: error: --target-features cannot be used with --march or --mcpu

# CHECK_X86: target triple = "x86_64-unknown-linux-gnu"
# CHECK_X86: "target-cpu"="haswell"

# CHECK_AARCH64: target triple = "aarch64-unknown-linux-gnu"
# CHECK_AARCH64: "target-cpu"="cortex-a72"

# CHECK_MARCH: target triple = "x86_64-unknown-linux-gnu"
# CHECK_MARCH: "target-cpu"="x86-64"


def main():
    pass
