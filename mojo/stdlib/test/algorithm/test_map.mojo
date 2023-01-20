# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_map::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import Buffer
from Int import Int
from IO import print
from Functional import map

# CHECK-LABEL: test_map
fn test_map():
    print("== test_map\n")

    # Create a mem of size 16
    let buffer = __mlir_op.`pop.stack_allocation`[
        count:5, _type : __mlir_type.`!pop.pointer<scalar<f32>>`
    ]()
    let vector = Buffer[
        5,
        __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
    ](buffer)

    vector.__setitem__(0, 1.0)
    vector.__setitem__(1, 2.0)
    vector.__setitem__(2, 3.0)
    vector.__setitem__(3, 4.0)
    vector.__setitem__(4, 5.0)

    @always_inline
    fn add_two(idx: Int):
        vector.__setitem__(idx, vector.__getitem__(idx) + 2)

    map[add_two](vector.__len__())

    # CHECK: 3.00
    print(vector.__getitem__(0))
    # CHECK: 4.00
    print(vector.__getitem__(1))
    # CHECK: 5.00
    print(vector.__getitem__(2))
    # CHECK: 6.00
    print(vector.__getitem__(3))
    # CHECK: 7.00
    print(vector.__getitem__(4))

    @always_inline
    fn add(idx: Int):
        vector.__setitem__(
            idx, vector.__getitem__(idx) + vector.__getitem__(idx)
        )

    map[add](vector.__len__())

    # CHECK: 6.00
    print(vector.__getitem__(0))
    # CHECK: 8.00
    print(vector.__getitem__(1))
    # CHECK: 10.00
    print(vector.__getitem__(2))
    # CHECK: 12.00
    print(vector.__getitem__(3))
    # CHECK: 14.00
    print(vector.__getitem__(4))


@export
fn main() -> __mlir_type.index:
    test_map()
    return 0
