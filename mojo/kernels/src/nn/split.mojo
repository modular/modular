# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param_msg, debug_assert
from Buffer import Buffer, DynamicRankBuffer
from DType import DType
from Index import product
from List import Dim, VariadicList
from Pointer import DTypePointer
from Memory import memcpy
from Range import range


# ===----------------------------------------------------------------------===#
# split
# ===----------------------------------------------------------------------===#


fn _split[
    type: DType
](
    input: Buffer[Dim(), type],
    axis: Int,
    outputs: VariadicList[DynamicRankBuffer],
):
    """splits input along axis and store in outputs.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. output i has dims [h, wi, c]. The input has dims [h, sum(wi), c] where
    i ranges from [0, num_outputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    let rank = outputs[0].rank
    let h = product(outputs[0].shape, 0, axis)
    let c = product(outputs[0].shape, axis + 1, rank)

    var w_in: Int = 0
    for ii in range(outputs.__len__()):
        w_in += outputs[ii].dim(axis)

    let stride_h_in = w_in * c
    let stride_w_in = c

    var w_offset: Int = 0
    for i in range(outputs.__len__()):
        # copy one w x c slice along h at a time
        let w = outputs[i].dim(axis)
        let out_buf = outputs[i].to_buffer[type]()
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
    type: DType, axis: Int
](input: Buffer[Dim(), type], outputs: VariadicList[DynamicRankBuffer],):
    assert_param_msg[axis == 0, "_split_inner only supports axis 0"]()
    var num_elems_copied: Int = 0
    for i in range(outputs.__len__()):
        let output_buf = outputs[i].to_buffer[type]()
        let buffer_len = output_buf.__len__()
        let input_buffer_offset = Buffer[Dim(), type](
            input.data.offset(num_elems_copied), buffer_len
        )
        memcpy[type](output_buf, input_buffer_offset)
        num_elems_copied += buffer_len


fn split[
    type: DType
](
    input: Buffer[Dim(), type],
    axis: Int,
    outputs: VariadicList[DynamicRankBuffer],
):
    # check inputs have same rank and same dims except for axis dim
    for i in range(outputs.__len__()):
        debug_assert(
            outputs[0].rank == outputs[i].rank,
            "all split inputs must have the same rank",
        )
        for j in range(outputs[i].rank):
            debug_assert(
                j == axis or outputs[0].dim(j) == outputs[i].dim(j),
                (
                    "all split outputs must have the same dimensions in the"
                    " non-split axes"
                ),
            )

    if axis == 0:
        _split_inner[type, 0](input, outputs)
        return

    _split[type](input, axis, outputs)
