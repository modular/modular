# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -emit-llvm %s -o %t.ll
# RUN: llvm-module-split %t.ll --per-func | FileCheck %s

# CHECK: # [LLVM Module Split: submodule 0]
# CHECK: @llvm.global_ctors
# CHECK: declare void @KGEN_EE_JIT_GlobalConstructor()

# CHECK: # [LLVM Module Split: submodule 1]
# CHECK: @llvm.global_dtors
# CHECK: declare void @KGEN_EE_JIT_GlobalDestructor()

# CHECK: # [LLVM Module Split: submodule 2]
# CHECK: define weak void @KGEN_EE_JIT_GlobalConstructor()

# CHECK: # [LLVM Module Split: submodule 3]
# CHECK: define weak void @KGEN_EE_JIT_GlobalDestructor()

# CHECK: # [LLVM Module Split: submodule 4]
# CHECK: define weak dso_local void @foo_closure_1(ptr %0)

# CHECK: # [LLVM Module Split: submodule 5]
# CHECK: @foo_async_closure_0_afp
# CHECK: define internal void @foo_async_closure_0_af(ptr %0)
# CHECK: define internal ptr @foo_async_closure_0()
# CHECK: declare dso_local void @foo_closure_1(ptr)
# CHECK: define dso_local void @foo()
# CHECK: define internal void @__kgen_coro_end_fn(ptr %0)


async fn but_async(b: Int) -> Int:
    return b + 2


@export
fn foo():
    var coro = but_async(1)
    _ = coro._deprecated_direct_resume()
