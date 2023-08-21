# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# FIXME(#18257): Flaky LSAN crashes.
# UNSUPPORTED: asan

# TODO(#19566): Reenable compilation with `-debug-level full`
# RUN: %mojo %s | FileCheck %s

from memory.buffer import Buffer, NDBuffer
from utils.index import Index, StaticIntTuple
from runtime.llcl import Runtime, OutputChainPtr, OwningOutputChainPtr
from utils.list import DimList
from algorithm import (
    all_true,
    any_true,
    mean,
    none_true,
    product,
    sum,
    variance,
    argmax,
    argmin,
)

# TODO: Fold this import into the one above.
from algorithm.reduction import max, min


# CHECK-LABEL: test_reductions
fn test_reductions():
    print("== test_reductions")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 1.0
    print(min(vector))

    # CHECK: 100.0
    print(max(vector))

    # CHECK: 5050.0
    print(sum(vector))


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
fn test_product():
    print("== test_product")

    alias simd_width = 4
    alias size = 10

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 3628800.0
    print(product(vector))


# CHECK-LABEL: test_mean_variance
fn test_mean_variance():
    print("== test_mean_variance")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 50.5
    print(mean(vector))

    # CHECK: 841.666687
    print(variance(vector, 1))


fn test_3d_reductions():
    print("== test_3d_reductions")
    alias simd_width = 4

    @always_inline
    @parameter
    fn _test_3d_reductions[
        input_shape: DimList,
        output_shape: DimList,
        reduce_axis: Int,
    ]():
        let input = NDBuffer[3, input_shape, DType.float32].stack_allocation()
        let output = NDBuffer[3, output_shape, DType.float32].stack_allocation()
        output.fill(0)

        for i in range(input.size()):
            input.flatten()[i] = i

        sum[
            3,
            input_shape,
            output_shape,
            DType.float32,
            reduce_axis,
        ](input, output)

        for i in range(output.size()):
            print(output.flatten()[i])

    # CHECK: 6.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 38.0
    # CHECK-NEXT: 54.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(2, 2, 1),
        2,
    ]()
    # CHECK: 4.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 24.0
    # CHECK-NEXT: 26.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(2, 1, 4),
        1,
    ]()
    # CHECK: 8.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 16.0
    # CHECK-NEXT: 18.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 22.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(1, 2, 4),
        0,
    ]()


# CHECK-LABEL: test_boolean
fn test_boolean():
    print("== test_boolean")

    alias simd_width = 2
    alias size = 5

    # Create a mem of size size
    let vector = Buffer[size, DType.bool].stack_allocation()
    vector[0] = True
    vector[1] = False
    vector[2] = False
    vector[3] = False
    vector[4] = True

    # CHECK: False
    print(all_true(vector))

    # CHECK: True
    print(any_true(vector))

    # CHECK: False
    print(none_true(vector))

    ###################################################
    # Check with all the elements set to True
    ###################################################

    for i in range(size):
        vector[i] = True.value

    # CHECK: True
    print(all_true(vector))

    # CHECK: True
    print(any_true(vector))

    # CHECK: False
    print(none_true(vector))

    ###################################################
    # Check with all the elements set to False
    ###################################################

    for i in range(size):
        vector[i] = False.value

    # CHECK: False
    print(all_true(vector))

    # CHECK: False
    print(any_true(vector))

    # CHECK: True
    print(none_true(vector))


# CHECK-LABEL: test_argn
fn test_argn():
    print("== test_argn")

    alias size = 15

    let vector = NDBuffer[1, DimList(size), DType.float32].stack_allocation()
    let output = NDBuffer[1, DimList(1), DType.index].stack_allocation()

    for i in range(size):
        vector[i] = i

    with Runtime(4) as runtime:
        let out_chain_0 = OwningOutputChainPtr(runtime)
        argmax(
            rebind[NDBuffer[1, DimList.create_unknown[1](), DType.float32]](
                vector
            ),
            0,
            rebind[NDBuffer[1, DimList.create_unknown[1](), DType.index]](
                output
            ),
            out_chain_0.borrow(),
        )
        out_chain_0.wait()
        # CHECK: argmax = 14
        print("argmax = ", output[0])

        let out_chain_1 = OwningOutputChainPtr(runtime)
        argmin(
            rebind[NDBuffer[1, DimList.create_unknown[1](), DType.float32]](
                vector
            ),
            0,
            rebind[NDBuffer[1, DimList.create_unknown[1](), DType.index]](
                output
            ),
            out_chain_1.borrow(),
        )
        out_chain_1.wait()
        # CHECK: argmin = 0
        print("argmin = ", output[0])


# CHECK-LABEL: test_argn_2
fn test_argn_2():
    print("== test_argn_2")

    alias batch_size = 4
    alias size = 15

    let vector = NDBuffer[
        2, DimList(batch_size, size), DType.float32
    ].stack_allocation()
    let output = NDBuffer[
        2, DimList(batch_size, 1), DType.index
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = i * size + j

    with Runtime(4) as runtime:
        let out_chain_0 = OwningOutputChainPtr(runtime)
        argmax(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_0.borrow(),
        )
        out_chain_0.wait()
        # CHECK: argmax = 14
        # CHECK: argmax = 14
        # CHECK: argmax = 14
        # CHECK: argmax = 14
        for i in range(batch_size):
            print("argmax = ", output[Index(i, 0)])

        let out_chain_1 = OwningOutputChainPtr(runtime)
        argmin(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_1.borrow(),
        )
        out_chain_1.wait()
        # CHECK: argmin = 0
        # CHECK: argmin = 0
        # CHECK: argmin = 0
        # CHECK: argmin = 0
        for i in range(batch_size):
            print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_2_test_2
fn test_argn_2_test_2():
    print("== test_argn_2_test_2")

    alias batch_size = 2
    alias size = 3

    let vector = NDBuffer[
        2, DimList(batch_size, size), DType.float32
    ].stack_allocation()
    let output = NDBuffer[
        2, DimList(batch_size, 1), DType.index
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = i * size + j
            if i % 2:
                vector[Index(i, j)] *= -1

    with Runtime(4) as runtime:
        let out_chain_0 = OwningOutputChainPtr(runtime)
        argmax(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_0.borrow(),
        )
        out_chain_0.wait()
        # CHECK: argmax = 2
        # CHECK: argmax = 0
        for i in range(batch_size):
            print("argmax = ", output[Index(i, 0)])

        let out_chain_1 = OwningOutputChainPtr(runtime)
        argmin(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_1.borrow(),
        )
        out_chain_1.wait()
        # CHECK: argmin = 0
        # CHECK: argmin = 2
        for i in range(batch_size):
            print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_2_neg_axis
fn test_argn_2_neg_axis():
    print("== test_argn_2_neg_axis")

    alias batch_size = 2
    alias size = 3

    let vector = NDBuffer[
        2, DimList(batch_size, size), DType.float32
    ].stack_allocation()
    let output = NDBuffer[
        2, DimList(batch_size, 1), DType.index
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = i * size + j
            if i % 2:
                vector[Index(i, j)] *= -1

    with Runtime(4) as runtime:
        let out_chain_0 = OwningOutputChainPtr(runtime)
        argmax(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            -1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_0.borrow(),
        )
        out_chain_0.wait()
        # CHECK: argmax = 2
        # CHECK: argmax = 0
        for i in range(batch_size):
            print("argmax = ", output[Index(i, 0)])

        let out_chain_1 = OwningOutputChainPtr(runtime)
        argmin(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            -1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_1.borrow(),
        )
        out_chain_1.wait()
        # CHECK: argmin = 0
        # CHECK: argmin = 2
        for i in range(batch_size):
            print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_test_zeros
fn test_argn_test_zeros():
    print("== test_argn_test_zeros")

    alias batch_size = 1
    alias size = 16

    let vector = NDBuffer[
        2, DimList(batch_size, size), DType.float32
    ].stack_allocation()
    let output = NDBuffer[
        2, DimList(batch_size, 1), DType.index
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = 0

    with Runtime(4) as runtime:
        let out_chain_0 = OwningOutputChainPtr(runtime)
        argmax(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_0.borrow(),
        )
        out_chain_0.wait()
        # CHECK: argmax = 0
        for i in range(batch_size):
            print("argmax = ", output[Index(i, 0)])

        let out_chain_1 = OwningOutputChainPtr(runtime)
        argmin(
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.float32]](
                vector
            ),
            1,
            rebind[NDBuffer[2, DimList.create_unknown[2](), DType.index]](
                output
            ),
            out_chain_1.borrow(),
        )
        out_chain_1.wait()
        # CHECK: argmin = 0
        for i in range(batch_size):
            print("argmin = ", output[Index(i, 0)])


fn main():
    test_reductions()
    test_product()
    test_mean_variance()
    test_3d_reductions()
    test_boolean()
    test_argn()
    test_argn_2()
    test_argn_2_test_2()
    test_argn_2_neg_axis()
    test_argn_test_zeros()
