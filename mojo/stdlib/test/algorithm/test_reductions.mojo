# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm import (
    all_true,
    any_true,
    argmax,
    argmin,
    cumsum,
    mean,
    none_true,
    product,
    sum,
    variance,
)
from algorithm.reduction import _reduce_generator, max, min
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from builtin.math import max as _max
from builtin.math import min as _min
from memory import UnsafePointer

from utils.index import Index, IndexList, StaticTuple


# CHECK-LABEL: test_reductions
fn test_reductions() raises:
    print("== test_reductions")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    var vector = Buffer[DType.float32, size].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 1.0
    print(min(vector))

    # CHECK: 100.0
    print(max(vector))

    # CHECK: 5050.0
    print(sum(vector))


# CHECK-LABEL: test_fused_reductions_inner
fn test_fused_reductions_inner() raises:
    print("== test_fused_redtest_fused_reductions_inneructions")

    alias size = 100
    alias test_type = DType.float32
    alias num_reductions = 3
    var vector = Buffer[test_type, size].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn input_fn[
        type: DType, width: Int, rank: Int
    ](indices: IndexList[rank]) -> SIMD[type, width]:
        var loaded_val = vector.load[width=width](indices[0])
        return rebind[SIMD[type, width]](loaded_val)

    var out = StaticTuple[Scalar[test_type], num_reductions]()

    @always_inline
    @parameter
    fn output_fn[
        type: DType, width: Int, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[type, width], num_reductions],
    ):
        constrained[
            width == 1,
            "Cannot write output if width is not equal to 1",
        ]()

        out = rebind[StaticTuple[Scalar[test_type], num_reductions]](val)

    @always_inline
    @parameter
    fn reduce_fn[
        ty: DType,
        width: Int,
        reduction_idx: Int,
    ](left: SIMD[ty, width], right: SIMD[ty, width],) -> SIMD[ty, width]:
        constrained[reduction_idx < num_reductions, "reduction_idx OOB"]()

        @parameter
        if reduction_idx == 0:
            return _min(left, right)
        elif reduction_idx == 1:
            return _max(left, right)
        else:
            return left + right

    var init_min = Scalar[test_type].MAX
    var init_max = Scalar[test_type].MIN
    var init = StaticTuple[Scalar[test_type], num_reductions](
        init_min, init_max, 0
    )
    var shape = Index(size)

    _reduce_generator[
        num_reductions, test_type, input_fn, output_fn, reduce_fn
    ](
        shape,
        init=init,
        reduce_dim=0,
    )

    # CHECK: 1.0
    print(out[0])

    # CHECK: 100.0
    print(out[1])

    # CHECK: 5050.0
    print(out[2])


# CHECK-LABEL: test_fused_reductions_outer
fn test_fused_reductions_outer() raises:
    print("== test_fused_reductions_outer")

    alias size = 100
    alias test_type = DType.float32
    alias num_reductions = 3
    var vector = Buffer[test_type, size].stack_allocation()

    # COM: For the purposes of this test, we reinterpret this as a tensor
    # COM: of shape [50, 2] and reduce along the outer dimension.
    # COM: A slice of the first column gives all odd numbers: 1, 3, 5 ... 99
    # COM: while a slice of the second gives all even numbers: 2, 4, 6, ... 100
    for i in range(size):
        vector[i] = i + 1

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn input_fn[
        type: DType, width: Int, rank: Int
    ](indices: IndexList[rank]) -> SIMD[type, width]:
        var loaded_val = vector.load[width=width](indices[0] * 2 + indices[1])
        return rebind[SIMD[type, width]](loaded_val)

    var col0 = StaticTuple[Scalar[test_type], num_reductions]()
    var col1 = StaticTuple[Scalar[test_type], num_reductions]()

    @always_inline
    @parameter
    fn reduce_fn[
        ty: DType,
        width: Int,
        reduction_idx: Int,
    ](left: SIMD[ty, width], right: SIMD[ty, width],) -> SIMD[ty, width]:
        constrained[reduction_idx < num_reductions, "reduction_idx OOB"]()

        @parameter
        if reduction_idx == 0:
            return _min(left, right)
        elif reduction_idx == 1:
            return _max(left, right)
        else:
            return left + right

    var init_min = Scalar[test_type].MAX
    var init_max = Scalar[test_type].MIN
    var init = StaticTuple[Scalar[test_type], num_reductions](
        init_min, init_max, 0
    )
    var shape = IndexList[2](50, 2)

    @always_inline
    @parameter
    fn output_fn[
        type: DType, width: Int, rank: Int
    ](
        indices: IndexList[rank],
        val: StaticTuple[SIMD[type, width], num_reductions],
    ):
        # CHECK: Column: 0  min:  1.0  max:  99.0  sum:  2500.0
        # CHECK: Column: 1  min:  2.0  max:  100.0  sum:  2550.0
        print(
            "Column:",
            indices[1],
            " min: ",
            val[0],
            " max: ",
            val[1],
            " sum: ",
            val[2],
        )

    _reduce_generator[
        num_reductions, test_type, input_fn, output_fn, reduce_fn
    ](
        shape,
        init=init,
        reduce_dim=0,
    )


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
fn test_product() raises:
    print("== test_product")

    alias simd_width = 4
    alias size = 10

    # Create a mem of size size
    var vector = Buffer[DType.float32, size].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 3628800.0
    print(product(vector))


# CHECK-LABEL: test_mean_variance
fn test_mean_variance() raises:
    print("== test_mean_variance")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    var vector = Buffer[DType.float32, size].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 50.5
    print(mean(vector))

    # CHECK: 841.666687
    print(variance(vector, 1))


@always_inline
@parameter
fn _test_3d_reductions[
    input_shape: DimList,
    output_shape: DimList,
    reduce_axis: Int,
]() raises:
    print("== test_3d_reductions reduce_axis=", reduce_axis)
    alias simd_width = 4
    var input = NDBuffer[DType.float32, 3, input_shape].stack_allocation()
    var output = NDBuffer[DType.float32, 3, output_shape].stack_allocation()
    output.fill(0)

    for i in range(input.size()):
        input.flatten()[i] = i

    sum[reduce_axis](input, output)

    for i in range(output.size()):
        print(output.flatten()[i])


# CHECK-LABEL: test_3d_reductions reduce_axis= 0
fn test_3d_reductions_axis_0() raises:
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


# CHECK-LABEL: test_3d_reductions reduce_axis= 1
fn test_3d_reductions_axis_1() raises:
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


# CHECK-LABEL: test_3d_reductions reduce_axis= 2
fn test_3d_reductions_axis_2() raises:
    # CHECK: 6.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 38.0
    # CHECK-NEXT: 54.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(2, 2, 1),
        2,
    ]()


# CHECK-LABEL: test_boolean
fn test_boolean():
    print("== test_boolean")

    alias simd_width = 2
    alias size = 5

    # Create a mem of size size
    var vector = Buffer[DType.bool, size].stack_allocation()
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
        vector[i] = True

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
        vector[i] = False

    # CHECK: False
    print(all_true(vector))

    # CHECK: False
    print(any_true(vector))

    # CHECK: True
    print(none_true(vector))


# CHECK-LABEL: test_argn
fn test_argn() raises:
    print("== test_argn")

    alias size = 93

    var vector = NDBuffer[DType.int32, 1, DimList(size)].stack_allocation()
    var output = NDBuffer[DType.index, 1, DimList(1)].stack_allocation()

    for i in range(size):
        vector[i] = i

    argmax(
        rebind[NDBuffer[DType.int32, 1]](vector),
        0,
        rebind[NDBuffer[DType.index, 1]](output),
    )

    # CHECK: argmax = 92
    print("argmax = ", output[0])

    argmin(
        rebind[NDBuffer[DType.int32, 1]](vector),
        0,
        rebind[NDBuffer[DType.index, 1]](output),
    )

    # CHECK: argmin = 0
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_2
fn test_argn_2() raises:
    print("== test_argn_2")

    alias batch_size = 4
    alias size = 91

    var vector = NDBuffer[
        DType.float32, 2, DimList(batch_size, size)
    ].stack_allocation()
    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = j

    argmax(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    for i in range(batch_size):
        print("argmax = ", output[Index(i, 0)])

    argmin(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_2_test_2
fn test_argn_2_test_2() raises:
    print("== test_argn_2_test_2")

    alias batch_size = 2
    alias size = 3

    var vector = NDBuffer[
        DType.float32, 2, DimList(batch_size, size)
    ].stack_allocation()
    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = i * size + j
            if i % 2:
                vector[Index(i, j)] *= -1

    argmax(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[Index(i, 0)])

    argmin(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_2_neg_axis
fn test_argn_2_neg_axis() raises:
    print("== test_argn_2_neg_axis")

    alias batch_size = 2
    alias size = 3

    var vector = NDBuffer[
        DType.float32, 2, DimList(batch_size, size)
    ].stack_allocation()
    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = i * size + j
            if i % 2:
                vector[Index(i, j)] *= -1

    argmax(
        rebind[NDBuffer[DType.float32, 2]](vector),
        -1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[Index(i, 0)])

    argmin(
        rebind[NDBuffer[DType.float32, 2]](vector),
        -1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_test_zeros
fn test_argn_test_zeros() raises:
    print("== test_argn_test_zeros")

    alias batch_size = 1
    alias size = 16

    var vector = NDBuffer[
        DType.float32, 2, DimList(batch_size, size)
    ].stack_allocation()
    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = 0

    argmax(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[Index(i, 0)])

    argmin(
        rebind[NDBuffer[DType.float32, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_test_identity
fn test_argn_test_identity() raises:
    print("== test_argn_test_identity")

    alias batch_size = 3
    alias size = 5

    var vector = NDBuffer[
        DType.int64, 2, DimList(batch_size, size)
    ].stack_allocation()
    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    for i in range(batch_size):
        for j in range(size):
            vector[Index(i, j)] = 0

    vector[Index(1, 4)] = 1
    vector[Index(2, 3)] = 1
    vector[Index(2, 4)] = 1

    argmax(
        rebind[NDBuffer[DType.int64, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 0
    print("argmax = ", output[Index(0, 0)])
    # CHECK: argmax = 4
    print("argmax = ", output[Index(1, 0)])
    # CHECK: argmax = 3
    print("argmax = ", output[Index(2, 0)])

    argmin(
        rebind[NDBuffer[DType.int64, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[Index(i, 0)])


# CHECK-LABEL: test_argn_3d_identity
fn test_argn_3d_identity() raises:
    print("== test_argn_3d_identity")

    alias batch_size = 2
    alias seq_len = 2
    alias hidden_dim = 5

    var vector = NDBuffer[
        DType.int64, 3, DimList(batch_size, seq_len, hidden_dim)
    ].stack_allocation()
    vector.fill(0)

    var output = NDBuffer[
        DType.index, 3, DimList(batch_size, seq_len, 1)
    ].stack_allocation()
    output.fill(0)

    vector[Index(0, 1, 4)] = 1
    vector[Index(1, 0, 1)] = 1
    vector[Index(1, 0, 2)] = 1
    vector[Index(1, 1, 3)] = 1

    argmax(
        rebind[NDBuffer[DType.int64, 3]](vector),
        2,
        rebind[NDBuffer[DType.index, 3]](output),
    )

    # CHECK: argmax = 0
    print("argmax = ", output[Index(0, 0, 0)])
    # CHECK: argmax = 4
    print("argmax = ", output[Index(0, 1, 0)])
    # CHECK: argmax = 1
    print("argmax = ", output[Index(1, 0, 0)])
    # CHECK: argmax = 3
    print("argmax = ", output[Index(1, 1, 0)])

    argmin(
        rebind[NDBuffer[DType.int64, 3]](vector),
        2,
        rebind[NDBuffer[DType.index, 3]](output),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        for j in range(seq_len):
            print("argmin = ", output[Index(i, j, 0)])


fn test_argn_less_than_simd() raises:
    print("== test_argn_less_than_simd")

    alias batch_size = 2
    alias hidden_dim = 3  # assumes simd_width of 4

    var vector = NDBuffer[
        DType.int64, 2, DimList(batch_size, hidden_dim)
    ].stack_allocation()
    vector.fill(0)

    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()
    output.fill(0)

    vector[Index(0, 0)] = 0
    vector[Index(0, 1)] = 1
    vector[Index(0, 2)] = 2
    vector[Index(1, 0)] = 5
    vector[Index(1, 1)] = 4
    vector[Index(1, 2)] = 3

    argmax(
        rebind[NDBuffer[DType.int64, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 2
    print("argmax = ", output[Index(0, 0)])
    # CHECK: argmax = 0
    print("argmax = ", output[Index(1, 0)])

    argmin(
        rebind[NDBuffer[DType.int64, 2]](vector),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 0
    print("argmin = ", output[Index(0, 0)])
    # CHECK: argmin = 2
    print("argmin = ", output[Index(1, 0)])


# CHECK-LABEL: test_argn_simd_edge_case
fn test_argn_simd_index_order() raises:
    print("== test_argn_simd_edge_case")

    # Checks the case where the maximal value is found in two simd_chunks, where
    # the index of the maximal value in the second simd_chunk is earlier than in the first.
    # ex:
    #   simd_width = 4
    #   [0, 0, 1, 0, 0, 1, 0, 0, 0]
    #   <--------->  <-------->  <>
    #          ^        ^
    alias size = 17

    var vector = NDBuffer[DType.int32, 1, DimList(size)].stack_allocation()
    vector.fill(0)
    var output = NDBuffer[DType.index, 1, DimList(1)].stack_allocation()

    vector[5] = 1
    vector[4] = -1
    vector[8] = -1
    vector[9] = 1

    argmax(
        rebind[NDBuffer[DType.int32, 1]](vector),
        0,
        rebind[NDBuffer[DType.index, 1]](output),
    )

    # CHECK: argmax = 5
    print("argmax = ", output[0])

    argmin(
        rebind[NDBuffer[DType.int32, 1]](vector),
        0,
        rebind[NDBuffer[DType.index, 1]](output),
    )

    # CHECK: argmin = 4
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_parallelize
fn test_argn_parallelize() raises:
    print("== test_argn_parallelize")

    # Checks argn's performance when the size of the NDBuffer exceeds the threshold to enable parallelism
    alias batch_size = 8
    alias hidden_dim = 16384

    var input_ptr = UnsafePointer[Float32].alloc(batch_size * hidden_dim)
    var input = NDBuffer[DType.float32, 2, DimList(batch_size, hidden_dim)](
        input_ptr
    )
    input.fill(0)

    var output = NDBuffer[
        DType.index, 2, DimList(batch_size, 1)
    ].stack_allocation()

    input[Index(0, 10)] = 100
    input[Index(0, 100)] = -100
    input[Index(1, 20)] = -100
    input[Index(1, 200)] = 100
    input[Index(2, 30)] = 100
    input[Index(2, 300)] = -100
    input[Index(3, 40)] = -100
    input[Index(3, 400)] = 100
    input[Index(4, 10)] = 100
    input[Index(4, 100)] = -100
    input[Index(5, 20)] = 100
    input[Index(5, 200)] = -100
    input[Index(6, 30)] = 100
    input[Index(6, 300)] = -100
    input[Index(7, 40)] = -100
    input[Index(7, 400)] = 100

    argmax(
        rebind[NDBuffer[DType.float32, 2]](input),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmax = 10
    print("argmax = ", output[Index(0, 0)])
    # CHECK: argmax = 200
    print("argmax = ", output[Index(1, 0)])
    # CHECK: argmax = 30
    print("argmax = ", output[Index(2, 0)])
    # CHECK: argmax = 400
    print("argmax = ", output[Index(3, 0)])
    # CHECK: argmax = 10
    print("argmax = ", output[Index(4, 0)])
    # CHECK: argmax = 20
    print("argmax = ", output[Index(5, 0)])
    # CHECK: argmax = 30
    print("argmax = ", output[Index(6, 0)])
    # CHECK: argmax = 400
    print("argmax = ", output[Index(7, 0)])

    argmin(
        rebind[NDBuffer[DType.float32, 2]](input),
        1,
        rebind[NDBuffer[DType.index, 2]](output),
    )

    # CHECK: argmin = 100
    print("argmin = ", output[Index(0, 0)])
    # CHECK: argmin = 20
    print("argmin = ", output[Index(1, 0)])
    # CHECK: argmin = 300
    print("argmin = ", output[Index(2, 0)])
    # CHECK: argmin = 40
    print("argmin = ", output[Index(3, 0)])
    # CHECK: argmin = 100
    print("argmin = ", output[Index(4, 0)])
    # CHECK: argmin = 200
    print("argmin = ", output[Index(5, 0)])
    # CHECK: argmin = 300
    print("argmin = ", output[Index(6, 0)])
    # CHECK: argmin = 40
    print("argmin = ", output[Index(7, 0)])

    input_ptr.free()


# CHECK-LABEL: test_cumsum
fn test_cumsum():
    print("== test_cumsum")

    var vector = Buffer[DType.float32, 150].stack_allocation()
    for i in range(len(vector)):
        vector[i] = i + 1
    var cumsum_out1 = Buffer[DType.float32, 150].stack_allocation()
    # cumsum[150, DType.float32](cumsum_out1, vector)
    # cumsum(cumsum_out1, vector)
    cumsum(cumsum_out1, vector)
    # CHECK: 1.0 ,3.0 ,6.0 ,10.0 ,15.0 ,21.0 ,28.0 ,36.0 ,45.0 ,55.0 ,66.0 ,78.0
    # CHECK: ,91.0 ,105.0 ,120.0 ,136.0 ,153.0 ,171.0 ,190.0 ,210.0 ,231.0
    # CHECK: ,253.0 ,276.0 ,300.0 ,325.0 ,351.0 ,378.0 ,406.0 ,435.0 ,465.0
    # CHECK: ,496.0 ,528.0 ,561.0 ,595.0 ,630.0 ,666.0 ,703.0 ,741.0 ,780.0
    # CHECK: ,820.0 ,861.0 ,903.0 ,946.0 ,990.0 ,1035.0 ,1081.0 ,1128.0 ,1176.0
    # CHECK: ,1225.0 ,1275.0 ,1326.0 ,1378.0 ,1431.0 ,1485.0 ,1540.0 ,1596.0
    # CHECK: ,1653.0 ,1711.0 ,1770.0 ,1830.0 ,1891.0 ,1953.0 ,2016.0 ,2080.0
    # CHECK: ,2145.0 ,2211.0 ,2278.0 ,2346.0 ,2415.0 ,2485.0 ,2556.0 ,2628.0
    # CHECK: ,2701.0 ,2775.0 ,2850.0 ,2926.0 ,3003.0 ,3081.0 ,3160.0 ,3240.0
    # CHECK: ,3321.0 ,3403.0 ,3486.0 ,3570.0 ,3655.0 ,3741.0 ,3828.0 ,3916.0
    # CHECK: ,4005.0 ,4095.0 ,4186.0 ,4278.0 ,4371.0 ,4465.0 ,4560.0 ,4656.0
    # CHECK: ,4753.0 ,4851.0 ,4950.0 ,5050.0 ,5151.0 ,5253.0 ,5356.0 ,5460.0
    # CHECK: ,5565.0 ,5671.0 ,5778.0 ,5886.0 ,5995.0 ,6105.0 ,6216.0 ,6328.0
    # CHECK: ,6441.0 ,6555.0 ,6670.0 ,6786.0 ,6903.0 ,7021.0 ,7140.0 ,7260.0
    # CHECK: ,7381.0 ,7503.0 ,7626.0 ,7750.0 ,7875.0 ,8001.0 ,8128.0 ,8256.0
    # CHECK: ,8385.0 ,8515.0 ,8646.0 ,8778.0 ,8911.0 ,9045.0 ,9180.0 ,9316.0
    # CHECK: ,9453.0 ,9591.0 ,9730.0 ,9870.0 ,10011.0 ,10153.0 ,10296.0 ,10440.0
    # CHECK: ,10585.0 ,10731.0 ,10878.0 ,11026.0 ,11175.0 ,11325.0 ,
    for i in range(cumsum_out1.__len__()):
        print(cumsum_out1[i], ",", end="")

    print()

    var vector2 = Buffer[DType.int64, 128].stack_allocation()
    for i in range(vector2.__len__()):
        vector2[i] = i + 1
    var cumsum_out2 = Buffer[DType.int64, 128].stack_allocation()
    # cumsum[128, DType.int64](cumsum_out2, vector2)
    # cumsum(cumsum_out2, vector2)
    cumsum(cumsum_out2, vector2)
    # CHECK: 1 ,3 ,6 ,10 ,15 ,21 ,28 ,36 ,45 ,55 ,66 ,78 ,91 ,105 ,120 ,136
    # CHECK: ,153 ,171 ,190 ,210 ,231 ,253 ,276 ,300 ,325 ,351 ,378 ,406 ,435
    # CHECK: ,465 ,496 ,528 ,561 ,595 ,630 ,666 ,703 ,741 ,780 ,820 ,861 ,903
    # CHECK: ,946 ,990 ,1035 ,1081 ,1128 ,1176 ,1225 ,1275 ,1326 ,1378 ,1431
    # CHECK: ,1485 ,1540 ,1596 ,1653 ,1711 ,1770 ,1830 ,1891 ,1953 ,2016 ,2080
    # CHECK: ,2145 ,2211 ,2278 ,2346 ,2415 ,2485 ,2556 ,2628 ,2701 ,2775 ,2850
    # CHECK: ,2926 ,3003 ,3081 ,3160 ,3240 ,3321 ,3403 ,3486 ,3570 ,3655 ,3741
    # CHECK: ,3828 ,3916 ,4005 ,4095 ,4186 ,4278 ,4371 ,4465 ,4560 ,4656 ,4753
    # CHECK: ,4851 ,4950 ,5050 ,5151 ,5253 ,5356 ,5460 ,5565 ,5671 ,5778 ,5886
    # CHECK: ,5995 ,6105 ,6216 ,6328 ,6441 ,6555 ,6670 ,6786 ,6903 ,7021 ,7140
    # CHECK: ,7260 ,7381 ,7503 ,7626 ,7750 ,7875 ,8001 ,8128 ,8256 ,
    for i in range(cumsum_out2.__len__()):
        print(cumsum_out2[i], ",", end="")


fn main() raises:
    test_reductions()
    test_fused_reductions_inner()
    test_fused_reductions_outer()
    test_product()
    test_mean_variance()
    test_3d_reductions_axis_0()
    test_3d_reductions_axis_1()
    test_3d_reductions_axis_2()
    test_boolean()
    test_argn()
    test_argn_2()
    test_argn_2_test_2()
    test_argn_2_neg_axis()
    test_argn_test_zeros()
    test_argn_test_identity()
    test_argn_3d_identity()
    test_argn_less_than_simd()
    test_argn_simd_index_order()
    test_argn_parallelize()
    test_cumsum()
