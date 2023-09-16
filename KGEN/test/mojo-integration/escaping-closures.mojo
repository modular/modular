# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn makes_escaping_closure(
    m: __mlir_type.index,
) -> fn (n: __mlir_type.index) escaping -> __mlir_type.index:
    fn myclosure(n: __mlir_type.index) escaping -> __mlir_type.index:
        return __mlir_op.`index.add`(n, m)

    return myclosure


fn main():
    let x = 2
    let c = makes_escaping_closure(x.value)
    # CHECK: 4
    print(c(x.value))
