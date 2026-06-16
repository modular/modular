# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: kgen -emit=llvm %s | FileCheck %s --check-prefix=IR

# Tests for __mlir_deferred_type: a return-type annotation that defers MLIR
# type construction (from a string template with parameter substitutions) until
# elaboration time.


@always_inline
def get_llvm_array[
    n: Int
]() -> __mlir_deferred_type[`!llvm.array<`, +n.__mlir_index__(), ` x f32>`]:
    return __mlir_op.`llvm.mlir.undef`[
        _type=__mlir_deferred_type[
            `!llvm.array<`, +n.__mlir_index__(), ` x f32>`
        ]
    ]()


# @no_inline forces the specialized function into LLVM IR so we can FileCheck
# that the deferred type resolved to [4 x float] and not some other size.
# IR: [4 x float]
@no_inline
def get_array_noinline[
    n: Int
]() -> __mlir_deferred_type[`!llvm.array<`, +n.__mlir_index__(), ` x f32>`]:
    return get_llvm_array[n]()


def main():
    # CHECK: ok
    comptime sz: Int = 4
    _ = get_array_noinline[sz]()
    print("ok")
