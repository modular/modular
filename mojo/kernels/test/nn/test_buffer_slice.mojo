# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import NDBuffer, Buffer, _raw_stack_allocation
from DType import DType
from Range import range
from DType import DType
from Functional import elementwise
from Int import Int
from Math import mul
from List import Dim, DimList, create_dim_list
from IO import print
from Index import StaticIntTuple, Index
from LLCL import Runtime, OwningOutputChainPtr
from TypeUtilities import rebind
from F32 import F32
from SIMD import SIMD
from Slice import slice_as_view, slice_as_copy


fn print_elements[
    type: DType, in_rank: Int
](tensor: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type]):
    print("New shape:")
    print(tensor.dynamic_shape)
    print("New strides:")
    print(tensor.dynamic_stride)

    @always_inline
    fn print_elements_lambda[
        simd_width: Int, rank: __mlir_type.index
    ](idx: StaticIntTuple[rank]):
        var index = rebind[StaticIntTuple[in_rank.__as_mlir_index()]](idx)
        print(tensor[index])

    let runtime = Runtime(1)
    let out_chain = OwningOutputChainPtr(runtime)

    elementwise[in_rank.__as_mlir_index(), 1, 1, print_elements_lambda](
        rebind[StaticIntTuple[in_rank.__as_mlir_index()]](tensor.dynamic_shape),
        out_chain.borrow(),
    )

    out_chain.wait()
    out_chain.__del__()
    runtime.__del__()


# slice_dim
fn test_slice[
    numelems: Int, outer_rank: Int, static_shape: DimList[outer_rank]
](
    dims: DimList[outer_rank],
    starts: StaticIntTuple[outer_rank.__as_mlir_index()],
    stops: StaticIntTuple[outer_rank.__as_mlir_index()],
    steps: StaticIntTuple[outer_rank.__as_mlir_index()],
    use_copy: Bool,
):

    # Isn't always used but is used for the output buffer if we copy.
    var output_mem = _raw_stack_allocation[numelems, DType.f32, 1]()

    var memory1 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var in_tensor = NDBuffer[
        outer_rank.__as_mlir_index(),
        rebind[DimList[outer_rank.__as_mlir_index()]](static_shape),
        DType.f32,
    ](
        memory1.address,
        dims,
        DType.f32,
    )

    print("In shape:")
    print(in_tensor.dynamic_shape)
    print("In strides:")
    print(in_tensor.dynamic_stride)

    var start_tensor_mem = _raw_stack_allocation[outer_rank, DType.index, 1]()
    var start_tensor = Buffer[Dim(), DType.index](
        start_tensor_mem.address, outer_rank
    )

    var end_tensor_mem = _raw_stack_allocation[outer_rank, DType.index, 1]()
    var end_tensor = Buffer[Dim(), DType.index](
        end_tensor_mem.address, outer_rank
    )

    var step_tensor_mem = _raw_stack_allocation[outer_rank, DType.index, 1]()
    var step_tensor = Buffer[Dim(), DType.index](
        step_tensor_mem.address, outer_rank
    )

    for dim in range(outer_rank):
        let start_val = SIMD[1, DType.index](starts[dim])
        start_tensor.data.offset(dim).store(start_val)

        let stop_val = SIMD[1, DType.index](stops[dim])
        end_tensor.data.offset(dim).store(stop_val)

        let step_val = SIMD[1, DType.index](steps[dim])
        step_tensor.data.offset(dim).store(step_val)

    var x: F32 = 0.0
    for i in range(numelems):
        in_tensor.data.offset(i).store(SIMD[1, DType.f32](x.value))
        x += 1.0

    # Perform the slice even if we are testing the copy so we get the target size.
    let sliced = slice_as_view[DType.f32, DType.index, outer_rank](
        rebind[
            NDBuffer[
                outer_rank, DimList[outer_rank].create_unknown(), DType.f32
            ]
        ](in_tensor),
        start_tensor,
        end_tensor,
        step_tensor,
    )

    if not use_copy:
        print_elements[DType.f32, outer_rank](sliced)
    else:
        print("As copy\n")

        var output_buffer = NDBuffer[
            outer_rank.__as_mlir_index(),
            rebind[DimList[outer_rank.__as_mlir_index()]](static_shape),
            DType.f32,
        ](
            output_mem.address,
            rebind[
                StaticIntTuple[
                    Int(outer_rank.__as_mlir_index()).__as_mlir_index()
                ]
            ](sliced.dynamic_shape),
            DType.f32,
        )

        let runtime = Runtime(1)
        let out_chain = OwningOutputChainPtr(runtime)
        slice_as_copy[DType.f32, DType.index, outer_rank](
            rebind[
                NDBuffer[
                    outer_rank, DimList[outer_rank].create_unknown(), DType.f32
                ]
            ](output_buffer),
            rebind[
                NDBuffer[
                    outer_rank, DimList[outer_rank].create_unknown(), DType.f32
                ]
            ](in_tensor),
            start_tensor,
            end_tensor,
            step_tensor,
            out_chain.borrow(),
        )
        out_chain.wait()
        out_chain.__del__()

        print_elements[DType.f32, outer_rank](
            rebind[
                NDBuffer[
                    outer_rank, DimList[outer_rank].create_unknown(), DType.f32
                ]
            ](output_buffer)
        )

        runtime.__del__()


# CHECK-LABEL: == test_slice_basic
fn test_slice_basic():
    print("== test_slice_basic\n")

    # CHECK-NEXT: In shape:(4, 4, 4)
    # CHECK-NEXT: In strides:(16, 4, 1)
    # CHECK-NEXT: New shape:(2, 2, 2)
    # CHECK-NEXT: New strides:(16, 4, 1)
    # CHECK-NEXT: [42.000000]
    # CHECK-NEXT: [43.000000]
    # CHECK-NEXT: [46.000000]
    # CHECK-NEXT: [47.000000]
    # CHECK-NEXT: [58.000000]
    # CHECK-NEXT: [59.000000]
    # CHECK-NEXT: [62.000000]
    # CHECK-NEXT: [63.000000]

    # print(torch.arange(0, 64).reshape(4, 4, 4)[2:4:1, 2:4:1, 2:4:1].flatten())
    test_slice[64, 3, DimList[3].create_unknown()](
        create_dim_list(4, 4, 4),
        Index(2, 2, 2),
        Index(4, 4, 4),
        Index(1, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_identity
fn test_slice_identity():
    print("== test_slice_identity\n")

    # CHECK-NEXT: In shape:(2, 2, 4)
    # CHECK-NEXT: In strides:(8, 4, 1)
    # CHECK-NEXT: New shape:(2, 2, 4)
    # CHECK-NEXT: New strides:(8, 4, 1)
    # CHECK-NEXT: [0.000000]
    # CHECK-NEXT: [1.000000]
    # CHECK-NEXT: [2.000000]
    # CHECK-NEXT: [3.000000]
    # CHECK-NEXT: [4.000000]
    # CHECK-NEXT: [5.000000]
    # CHECK-NEXT: [6.000000]
    # CHECK-NEXT: [7.000000]
    # CHECK-NEXT: [8.000000]
    # CHECK-NEXT: [9.000000]
    # CHECK-NEXT: [10.000000]
    # CHECK-NEXT: [11.000000]
    # CHECK-NEXT: [12.000000]
    # CHECK-NEXT: [13.000000]
    # CHECK-NEXT: [14.000000]
    # CHECK-NEXT: [15.000000]

    # print(torch.arange(0, 16).reshape(2, 2, 4)[0:2:1, 0:2:1, 0:4:1].flatten())

    # Check slicing along all dimensions returns the original tensor.
    test_slice[16, 3, DimList[3].create_unknown()](
        create_dim_list(2, 2, 4),
        Index(0, 0, 0),
        Index(2, 2, 4),
        Index(1, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_steps
fn test_slice_steps():
    print("== test_slice_steps\n")

    # CHECK-NEXT: In shape:(2, 4, 8)
    # CHECK-NEXT: In strides:(32, 8, 1)
    # CHECK-NEXT: New shape:(1, 2, 4)
    # CHECK-NEXT: New strides:(64, 16, 2)
    # CHECK-NEXT: [0.000000]
    # CHECK-NEXT: [2.000000]
    # CHECK-NEXT: [4.000000]
    # CHECK-NEXT: [6.000000]
    # CHECK-NEXT: [16.000000]
    # CHECK-NEXT: [18.000000]
    # CHECK-NEXT: [20.000000]
    # CHECK-NEXT: [22.000000]

    # print(torch.arange(0, 64).reshape(2, 4, 8)[0:2:2, 0:4:2, 0:8:2].flatten())
    test_slice[64, 3, DimList[3].create_unknown()](
        create_dim_list(2, 4, 8),
        Index(0, 0, 0),
        Index(2, 4, 8),
        Index(2, 2, 2),
        False,
    )


# CHECK-LABEL: == test_slice_1D
fn test_slice_1D():
    print("== test_slice_1D\n")

    # CHECK-NEXT: In shape:(64, )
    # CHECK-NEXT: In strides:(1, )
    # CHECK-NEXT: New shape:(4, )
    # CHECK-NEXT: New strides:(4, )
    # CHECK-NEXT: [16.000000]
    # CHECK-NEXT: [20.000000]
    # CHECK-NEXT: [24.000000]
    # CHECK-NEXT: [28.000000]

    # print(torch.arange(0, 64)[16:30:4].flatten())
    test_slice[64, 1, DimList[1].create_unknown()](
        create_dim_list(64), Index(16), Index(30), Index(4), False
    )


# CHECK-LABEL: == test_slice_empty
fn test_slice_empty():
    print("== test_slice_empty\n")

    # CHECK-NEXT: In shape:(64, )
    # CHECK-NEXT: In strides:(1, )
    # CHECK-NEXT: New shape:(0, )
    # CHECK-NEXT: New strides:(1, )

    # print(torch.arange(0, 64)[8:8:1].flatten())
    test_slice[64, 1, DimList[1].create_unknown()](
        create_dim_list(64), Index(8), Index(8), Index(1), False
    )


# CHECK-LABEL: == test_slice_4D
fn test_slice_4D():
    print("== test_slice_4D\n")

    # CHECK-NEXT: In shape:(2, 4, 4, 2)
    # CHECK-NEXT: In strides:(32, 8, 2, 1)
    # CHECK-NEXT: New shape:(1, 1, 4, 1)
    # CHECK-NEXT: New strides:(32, 16, 2, 1)
    # CHECK-NEXT: [49.000000]
    # CHECK-NEXT: [51.000000]
    # CHECK-NEXT: [53.000000]
    # CHECK-NEXT: [55.000000]

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[1:2:1, 2:4:2, 0:4:1, 1:2:1].flatten())
    test_slice[64, 4, DimList[4].create_unknown()](
        create_dim_list(2, 4, 4, 2),
        Index(1, 2, 0, 1),
        Index(2, 4, 4, 2),
        Index(1, 2, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_copy
fn test_slice_copy():
    print("== test_slice_copy\n")

    # CHECK-NEXT: In shape:(2, 4, 4, 2)
    # CHECK-NEXT: In strides:(32, 8, 2, 1)
    # CHECK-NEXT: As copy
    # CHECK-NEXT: New shape:(1, 1, 4, 1)

    # Strides should be contiguous in the copy.
    # CHECK-NEXT: New strides:(4, 4, 1, 1)
    # CHECK-NEXT: [49.000000]
    # CHECK-NEXT: [51.000000]
    # CHECK-NEXT: [53.000000]
    # CHECK-NEXT: [55.000000]

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[1:2:1, 2:4:2, 0:4:1, 1:2:1].flatten())
    test_slice[64, 4, DimList[4].create_unknown()](
        create_dim_list(2, 4, 4, 2),
        Index(1, 2, 0, 1),
        Index(2, 4, 4, 2),
        Index(1, 2, 1, 1),
        True,
    )


# CHECK-LABEL: == test_slice_negative
fn test_slice_negative():
    print("== test_slice_negative\n")

    # CHECK-NEXT: In shape:(2, 4, 4, 2)
    # CHECK-NEXT: In strides:(32, 8, 2, 1)
    # CHECK-NEXT: New shape:(1, 2, 4, 1)
    # CHECK-NEXT: New strides:(32, 16, 2, 1)

    # CHECK-NEXT: [1.000000]
    # CHECK-NEXT: [3.000000]
    # CHECK-NEXT: [5.000000]
    # CHECK-NEXT: [7.000000]

    # CHECK-NEXT: [17.000000]
    # CHECK-NEXT: [19.000000]
    # CHECK-NEXT: [21.000000]
    # CHECK-NEXT: [23.000000]

    # print(torch.arange(0, 64).reshape(2, 4, 4, 2)[-2:-1:1, -4:-1:2, -4:4:1, -1:2:1].flatten())
    test_slice[64, 4, DimList[4].create_unknown()](
        create_dim_list(2, 4, 4, 2),
        Index(-2, -4, -4, -1),
        Index(-1, -1, 4, 2),
        Index(1, 2, 1, 1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_1D
fn test_slice_negative_step_1D():
    print("== test_slice_negative_step_1D\n")

    # CHECK: In shape:(15, )
    # CHECK-NEXT: In strides:(1, )
    # CHECK-NEXT: New shape:(6, )
    # CHECK-NEXT: New strides:(-1, )

    # CHECK-NEXT: [14.000000]
    # CHECK-NEXT: [13.000000]
    # CHECK-NEXT: [12.000000]
    # CHECK-NEXT: [11.000000]
    # CHECK-NEXT: [10.000000]
    # CHECK-NEXT: [9.000000]

    # print(np.arange(0, 15)[14:8:-1])
    test_slice[15, 1, DimList[1].create_unknown()](
        create_dim_list(
            15,
        ),
        Index(
            14,
        ),
        Index(
            8,
        ),
        Index(
            -1,
        ),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_2D
fn test_slice_negative_step_2D():
    print("== test_slice_negative_step_2D\n")

    # CHECK: In shape:(16, 4)
    # CHECK-NEXT: In strides:(4, 1)
    # CHECK-NEXT: New shape:(4, 2)
    # CHECK-NEXT: New strides:(-8, -1)

    # CHECK-NEXT: [59.000000]
    # CHECK-NEXT: [58.000000]
    # CHECK-NEXT: [51.000000]
    # CHECK-NEXT: [50.000000]
    # CHECK-NEXT: [43.000000]
    # CHECK-NEXT: [42.000000]
    # CHECK-NEXT: [35.000000]
    # CHECK-NEXT: [34.000000]

    # print(np.arange(0, 64).reshape(16, 4)[14:6:-2, -1:1:-1])
    test_slice[64, 2, DimList[2].create_unknown()](
        create_dim_list(16, 4),
        Index(14, -1),
        Index(6, 1),
        Index(-2, -1),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_3D
fn test_slice_negative_step_3D():
    print("== test_slice_negative_step_3D\n")

    # CHECK: In shape:(8, 2, 4)
    # CHECK-NEXT: In strides:(8, 4, 1)
    # CHECK-NEXT: New shape:(2, 2, 2)
    # CHECK-NEXT: New strides:(-16, 4, -2)

    # CHECK-NEXT: [59.000000]
    # CHECK-NEXT: [57.000000]
    # CHECK-NEXT: [63.000000]
    # CHECK-NEXT: [61.000000]
    # CHECK-NEXT: [43.000000]
    # CHECK-NEXT: [41.000000]
    # CHECK-NEXT: [47.000000]
    # CHECK-NEXT: [45.000000]

    # print(np.arange(0, 64).reshape(8, 2, 4)[-1:4:-2, :, 4:0:-2])
    test_slice[64, 3, DimList[3].create_unknown()](
        create_dim_list(8, 2, 4),
        Index(-1, 0, -1),
        Index(4, 2, 0),
        Index(-2, 1, -2),
        False,
    )


# CHECK-LABEL: == test_slice_negative_step_4D
fn test_slice_negative_step_4D():
    print("== test_slice_negative_step_4D\n")

    # CHECK: In shape:(2, 4, 2, 4)
    # CHECK-NEXT: In strides:(32, 8, 4, 1)
    # CHECK-NEXT: New shape:(1, 2, 1, 3)
    # CHECK-NEXT: New strides:(-32, -16, -4, -1)

    # CHECK-NEXT: [63.000000]
    # CHECK-NEXT: [62.000000]
    # CHECK-NEXT: [61.000000]
    # CHECK-NEXT: [47.000000]
    # CHECK-NEXT: [46.000000]
    # CHECK-NEXT: [45.000000]

    # print(np.arange(0, 64).reshape(2, 4, 2, 4)[-1:0:-1, -1:0:-2, -1:0:-1, -1:0:-1].stride)
    test_slice[64, 4, DimList[4].create_unknown()](
        create_dim_list(2, 4, 2, 4),
        Index(-1, -1, -1, -1),
        Index(0, 0, 0, 0),
        Index(-1, -2, -1, -1),
        False,
    )


fn main():
    test_slice_basic()
    test_slice_identity()
    test_slice_steps()
    test_slice_1D()
    test_slice_empty()
    test_slice_4D()
    test_slice_copy()
    test_slice_negative()

    test_slice_negative_step_1D()
    test_slice_negative_step_2D()
    test_slice_negative_step_3D()
    test_slice_negative_step_4D()
