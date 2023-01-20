# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_list::main():index()' -I %stdlibdir | FileCheck %s

from IO import print
from List import create_kgen_list, product

# CHECK-LABEL: test_list
fn test_list():
    print("== test_list\n")

    let lst = create_kgen_list[__mlir_type.index](1, 2, 3, 4)

    # CHECK: [1, 2, 3, 4]
    print[
        4,
        __mlir_type.index,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](lst)

    # CHECK: 24
    print(product[4](lst))


@export
fn main() -> __mlir_type.index:
    test_list()
    return 0
