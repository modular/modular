# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


from math import iota
from memory.buffer import NDBuffer
from algorithm.reduction import _get_nd_indices_from_flat_index
from TopK import top_k


fn get_tensor_storage[
    rank: Int, type: DType
](shape: StaticIntTuple[rank]) -> DynamicVector[SIMD[type, 1]]:
    var ret = DynamicVector[SIMD[type, 1]](shape.flattened_length())
    ret.resize(shape.flattened_length())
    return ret


fn to_ndbuffer[
    rank: Int, type: DType
](
    storage: DynamicVector[SIMD[type, 1]], shape: StaticIntTuple[rank]
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    return NDBuffer[rank, DimList.create_unknown[rank](), type](
        rebind[DTypePointer[type]](storage.data), shape
    )


fn test_case[
    rank: Int,
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (
        inout NDBuffer[rank, DimList.create_unknown[rank](), type]
    ) capturing -> None,
](K: Int, axis: Int, input_shape: StaticIntTuple[rank], largest: Bool = True):
    var input_storage = get_tensor_storage[rank, type](input_shape)

    var output_shape = input_shape
    output_shape[axis] = K
    var out_vals_storage = get_tensor_storage[rank, type](output_shape)
    var out_idxs_storage = get_tensor_storage[rank, DType.int64](output_shape)

    alias unknown_shape = DimList.create_unknown[rank]()

    var input = to_ndbuffer[rank, type](input_storage, input_shape)
    fill_fn[rank, type](input)
    let output_vals = to_ndbuffer[rank, type](out_vals_storage, output_shape)
    let output_idxs = to_ndbuffer[rank, DType.int64](
        out_idxs_storage, output_shape
    )

    top_k(input, K, axis, largest, output_vals, output_idxs)

    for i in range(out_vals_storage.size):
        print_no_newline(out_vals_storage[i])
        print_no_newline(",")
    print("")
    for i in range(out_idxs_storage.size):
        print_no_newline(out_idxs_storage[i])
        print_no_newline(",")
    print("")

    input_storage._del_old()
    out_vals_storage._del_old()
    out_idxs_storage._del_old()


fn main():
    @parameter
    fn fill_iota[
        rank: Int, type: DType
    ](inout buf: NDBuffer[rank, DimList.create_unknown[rank](), type]):
        iota[type](buf.data, buf.get_shape().flattened_length())

    fn test_axis_1():
        print("== test_axis_1")
        test_case[2, DType.float32, fill_iota](2, 1, StaticIntTuple[2](4, 4))

    # CHECK-LABEL: test_axis_1
    # CHECK: 3.0,2.0,7.0,6.0,11.0,10.0,15.0,14.0,
    # CHECK-NEXT: 3,2,3,2,3,2,3,2,
    test_axis_1()

    fn test_smallest():
        print("== test_smallest")
        test_case[2, DType.float32, fill_iota](
            2, 1, StaticIntTuple[2](4, 4), False
        )

    # CHECK-LABEL: test_smallest
    # CHECK: 0.0,1.0,4.0,5.0,8.0,9.0,12.0,13.0,
    # CHECK-NEXT: 0,1,0,1,0,1,0,1,
    test_smallest()

    fn test_axis_0():
        print("== test_axis_0")
        test_case[2, DType.float32, fill_iota](2, 0, StaticIntTuple[2](4, 4))

    # CHECK-LABEL: test_axis_0
    # CHECK: 12.0,13.0,14.0,15.0,8.0,9.0,10.0,11.0,
    # CHECK-NEXT: 3,3,3,3,2,2,2,2,
    test_axis_0()

    @parameter
    fn fill_identical[
        rank: Int, type: DType
    ](inout buf: NDBuffer[rank, DimList.create_unknown[rank](), type]):
        buf.fill(1)

    fn test_identical():
        print("== test_identical")
        test_case[2, DType.float32, fill_identical](
            3, 0, StaticIntTuple[2](4, 4)
        )

    # CHECK-LABEL: test_identical
    # CHECK: 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # CHECK-NEXT: 0,0,0,0,1,1,1,1,2,2,2,2,
    test_identical()

    fn test_max_k():
        print("== test_max_k")
        test_case[2, DType.float32, fill_iota](3, 0, StaticIntTuple[2](3, 4))

    # CHECK-LABEL: test_max_k
    # CHECK: 8.0,9.0,10.0,11.0,4.0,5.0,6.0,7.0,0.0,1.0,2.0,3.0,
    # CHECK-NEXT: 2,2,2,2,1,1,1,1,0,0,0,0,
    test_max_k()

    @parameter
    fn fill_custom[
        rank: Int, type: DType
    ](inout buf: NDBuffer[rank, DimList.create_unknown[rank](), type]):
        var flat_buf = buf.flatten()
        for i in range(flat_buf.__len__()):
            flat_buf[i] = flat_buf.__len__() - i - 1
        flat_buf[0] = -1

    fn test_5d():
        print("== test_5d")
        test_case[5, DType.float32, fill_custom](
            1, 1, StaticIntTuple[5](1, 4, 3, 2, 1)
        )

    # CHECK-LABEL: == test_5d
    # CHECK: 17.0,22.0,21.0,20.0,19.0,18.0,
    # CHECK-NEXT: 1,0,0,0,0,0,
    test_5d()
