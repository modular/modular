# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_partial_load_store::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import (
    Buffer,
    NDBuffer,
    raw_stack_allocation,
    partial_simd_load,
    partial_simd_store,
)
from Int import Int
from Index import Index
from IO import print
from List import create_kgen_list
from Math import exp


# CHECK-LABEL: test_partial_load_store
fn test_partial_load_store():
    print("== test_partial_load_store\n")
    # The total amount of data to allocate
    alias total_buffer_size: __mlir_type.index = 32

    var read_data = raw_stack_allocation[
        total_buffer_size,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        1,
    ]()

    var write_data = raw_stack_allocation[
        total_buffer_size,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        1,
    ]()

    var read_buffer = Buffer[
        total_buffer_size,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](read_data.address)

    var write_buffer = Buffer[
        total_buffer_size,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](write_data.address)

    var idx: Int = 0
    while idx < total_buffer_size:
        # Fill read_bufer with 0->15
        read_buffer.__setitem__(idx, idx.__as_mlir_index())
        # Fill write_buffer with 0
        write_buffer.__setitem__(idx, 0)
        idx += 1

    # Test partial load:
    let partial_load_data = partial_simd_load[
        4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    ](
        read_buffer.data.offset(1),
        1,
        3,
        99,  # idx  # lbound  # rbound  # pad value
    )
    # CHECK: [99, 2, 3, 99]
    print[4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        partial_load_data
    )

    # Test partial store:
    partial_simd_store[
        4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    ](
        write_buffer.data.offset(1),
        2,
        4,
        partial_load_data,  # idx  # lbound  # rbound
    )
    let partial_store_data = write_buffer.simd_load[4](2)
    # CHECK: [0, 3, 99, 0]
    print[4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        partial_store_data
    )

    # Test NDBuffer partial load store
    var read_nd_buffer = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](8, 4),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](read_data.address)

    var write_nd_buffer = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](8, 4),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](write_data.address)

    # Test partial load:
    let nd_partial_load_data = partial_simd_load[
        4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    ](
        read_nd_buffer._offset(Index(3, 2).as_tuple()),
        0,
        2,
        123,  # lbound  # rbound  # pad value
    )
    # CHECK: [14, 15, 123, 123]
    print[4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        nd_partial_load_data
    )

    # Test partial store
    partial_simd_store[
        4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    ](
        write_nd_buffer._offset(Index(3, 1).as_tuple()),
        0,  # lbound
        3,  # rbound
        nd_partial_load_data,  # value
    )
    let nd_partial_store_data = write_nd_buffer.simd_load[4](
        Index(3, 0).as_tuple()
    )

    # CHECK: [0, 14, 15, 123]
    print[4, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        nd_partial_store_data
    )


@export
fn main() -> __mlir_type.index:
    test_partial_load_store()
    return 0
