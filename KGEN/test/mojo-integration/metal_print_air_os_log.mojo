# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: system-darwin
# RUN: mojo %s | FileCheck %s --dump-input=fail
#
# Verifies that `print(...)` on Apple GPU lowers to a direct `air.os_log`
# call expressed entirely in Mojo (no `__mojo_metal_os_log_64` sentinel
# and no LLVM pass-side rewrite).

from std.gpu.host.compile import _compile_code
from std.gpu.host import get_gpu_target


def metal_print_kernel():
    print("hello")


def main():
    print(
        _compile_code[
            metal_print_kernel,
            emission_kind="llvm",
            target=get_gpu_target["metal:3"](),
        ]()
    )


# The subsystem, category, and format strings must be materialized as
# `internal addrspace(2) constant` LLVM globals (via
# `pop.global_constant` with a non-zero address space). Constants stop
# `splitPerExported` from grouping the kernels that share them, which is
# why multi-kernel `print()` works end-to-end. Each constant is the
# `{ [N x i8] }` lowering of a Mojo `InlineArray[Int8, N]`.
# CHECK-DAG: internal addrspace(2) constant { [5 x i8] } { [5 x i8] c"mojo\00" }
# CHECK-DAG: internal addrspace(2) constant { [6 x i8] } { [6 x i8] c"print\00" }
# CHECK-DAG: internal addrspace(2) constant { [129 x i8] } { [129 x i8] c"%c{{.*}}\00" }

# 64 packed `%c` slots, 4 bytes each, must show up as a 64-element i32
# stack buffer (`stack_allocation[64, Int32]()` lowers to this form).
# CHECK: alloca i32, i64 64, align 4

# The call site must pass the constant subsystem/category/format strings
# in `addrspace(2)`, the literal log type 1, the generic-AS va buffer,
# and a va_size of 256 bytes (= 64 * 4).
# CHECK: call void @air.os_log(ptr addrspace(2) {{[^,]+}}, ptr addrspace(2) {{[^,]+}}, i32 1, ptr addrspace(2) {{[^,]+}}, ptr {{[^,]+}}, i64 256)

# The intrinsic itself must be declared with the exact signature
# `osLogFTy` expects: subsystem/category/format in `addrspace(2)`, log
# type i32, va-arg buffer in the generic address space, and va-size i64.
# CHECK: declare void @air.os_log(ptr addrspace(2), ptr addrspace(2), i32, ptr addrspace(2), ptr, i64)

# The sentinel function and its declaration must be gone end-to-end. The
# `addrspacecast`-and-clone-pass workaround must also be gone — the AS2
# globals are emitted directly via `pop.global_constant`, so no
# `addrspacecast` appears at the call site.
# CHECK-NOT: __mojo_metal_os_log_64
# CHECK-NOT: addrspacecast
