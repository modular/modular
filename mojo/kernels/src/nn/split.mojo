# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert
from Buffer import Buffer, NDBuffer, DynamicRankBuffer
from DType import DType
from Index import product
from Intrinsics import external_call
from List import Dim, VariadicList, DimList
from Memory import memcpy
from Range import range
from LLCL import OutputChainPtr
from Functional import sync_parallelize
from Vector import InlinedFixedVector


struct _NDBufferVector[rank: Int, type: DType]:
    """Utility to store a VariadicList of NDBuffers. Required because there is not
    a clean way to convert a VariadicList of DynamicRankBuffers to a VariadicList
    of NDBuffers."""

    alias stack_capacity = 20
    alias BufferType = NDBuffer[rank, DimList.create_unknown[rank](), type]
    alias StorageType = InlinedFixedVector[Self.stack_capacity, Self.BufferType]
    var storage: Self.StorageType

    @always_inline
    fn __init__(inout self, num_inputs: Int):
        self.storage = Self.StorageType(num_inputs)

    @always_inline
    fn __init__(inout self, *inputs: DynamicRankBuffer):
        self.__init__(VariadicList[DynamicRankBuffer](inputs))

    fn __init__(inout self, input_list: VariadicList[DynamicRankBuffer]):
        self.storage = Self.StorageType(input_list.__len__())
        for i in range(input_list.__len__()):
            self.storage.append(input_list[i].to_ndbuffer[rank, type]())

    @always_inline
    fn __init__(inout self, *inputs: Self.BufferType):
        self.__init__(VariadicList[Self.BufferType](inputs))

    fn __init__(inout self, input_list: VariadicList[Self.BufferType]):
        self.storage = Self.StorageType(input_list.__len__())
        for i in range(input_list.__len__()):
            self.storage.append(input_list[i])

    @always_inline
    fn __getitem__(
        self, idx: Int
    ) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
        return self.storage[idx]

    @always_inline
    fn __len__(self) -> Int:
        return self.storage.__len__()

    @always_inline
    fn __del__(owned self):
        return self.storage._del_old()


# ===----------------------------------------------------------------------===#
# split
# ===----------------------------------------------------------------------===#


fn _split[
    type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    outputs: _NDBufferVector[rank, type],
):
    """splits input along axis and store in outputs.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. output i has dims [h, wi, c]. The input has dims [h, sum(wi), c] where
    i ranges from [0, num_outputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    let shape = outputs[0].get_shape()
    let h = product(shape, 0, axis)
    let c = product(shape, axis + 1, rank)

    var w_in: Int = 0
    for i in range(outputs.__len__()):
        w_in += outputs[i].dim(axis)

    let stride_h_in = w_in * c
    let stride_w_in = c

    var w_offset: Int = 0
    for i in range(outputs.__len__()):
        # copy one w x c slice along h at a time
        let w = outputs[i].dim(axis)
        let out_buf = outputs[i].flatten()
        for j in range(h):
            let output_offset = j * w * c
            let input_offset = j * stride_h_in + w_offset * stride_w_in
            let out_slice = Buffer[Dim(), type](
                out_buf.data + output_offset, w * c
            )
            let in_slice = Buffer[Dim(), type](input.data + input_offset, w * c)
            # these slices are contiguous
            memcpy(out_slice, in_slice)
        w_offset += w


fn _split_inner[
    type: DType, rank: Int, axis: Int
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    outputs: _NDBufferVector[rank, type],
):
    assert_param[axis == 0, "_split_inner only supports axis 0"]()
    var num_elems_copied: Int = 0
    for i in range(outputs.__len__()):
        let output_buf = outputs[i].flatten()
        let buffer_len = output_buf.__len__()
        let input_buffer_offset = Buffer[Dim(), type](
            input.data.offset(num_elems_copied), buffer_len
        )
        memcpy[type](output_buf, input_buffer_offset)
        num_elems_copied += buffer_len


fn split[
    type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
    outputs: _NDBufferVector[rank, type],
    out_chain: OutputChainPtr,
):
    # check inputs have same rank and same dims except for axis dim
    for i in range(outputs.__len__()):
        if outputs[0].get_rank() != outputs[i].get_rank():
            return out_chain.mark_error(
                "all split inputs must have the same rank"
            )
        for j in range(outputs[i].get_rank()):
            if j != axis and outputs[0].dim(j) != outputs[i].dim(j):
                return out_chain.mark_error(
                    "all split outputs must have the same dimensions in the"
                    " non-split axes"
                )

    @parameter
    @always_inline
    fn task_func(task_id: Int):
        if axis == 0:
            _split_inner[type, rank, 0](input, outputs)
            return

        _split[type](input, axis, outputs)

    # The task_func closure captures the stack allocated _NDBufferVector,
    # so this kernel must run synchronously.
    sync_parallelize[task_func](out_chain, 1)
