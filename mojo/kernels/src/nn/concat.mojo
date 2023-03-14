# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param_bool_msg, debug_assert
from Buffer import Buffer, DynamicRankBuffer
from DType import DType
from Int import Int
from List import VariadicList
from Pointer import DTypePointer, product
from Memory import memcpy
from Range import range

# ===----------------------------------------------------------------------===#
# concat
# ===----------------------------------------------------------------------===#

alias MAX_RANK = 5
alias unknown = __mlir_attr.`#kgen.unknown : index`


fn _concat[
    type: DType
](
    output: Buffer[__mlir_attr.`#kgen.unknown : index`, type],
    axis: Int,
    inputs: VariadicList[DynamicRankBuffer],
):
    """Concatenate inputs along axis and store in output.

    This simplifies the implementation by reshaping the output and inputs into 3D
    buffers. input i has dims [h, wi, c]. The output has dims [h, sum(wi), c] where
    i ranges from [0, num_inputs).

    Reshaping the buffer does not change the memory layout. After reshaping to 3D
    it is easy to visualize that the inputs can be copied in w x c sized
    contiguous slices along the h dimension.

    """

    let rank = inputs[0].rank
    let h = product(inputs[0].shape, 0, axis)
    let c = product(inputs[0].shape, axis + 1, rank)

    var w_out: Int = 0
    for ii in range(inputs.__len__()):
        w_out += inputs[ii].shape.load(axis).value

    let stride_h_out = w_out * c
    let stride_w_out = c

    var w_offset: Int = 0
    for i in range(inputs.__len__()):
        # copy one w x c slice along h at a time
        let w = inputs[i].shape.load(axis).value
        let in_buf = inputs[i].to_buffer[type]()
        for j in range(h):
            let input_offset = j * w * c
            let output_offset = j * stride_h_out + w_offset * stride_w_out
            let in_slice = Buffer[unknown, type](
                in_buf.data + input_offset, w * c
            )
            let out_slice = Buffer[unknown, type](
                output.data + output_offset, w * c
            )
            # these slices are contiguous
            memcpy(out_slice, in_slice)
        w_offset += w


fn _concat_inner[
    type: DType, axis: Int
](output: Buffer[unknown, type], inputs: VariadicList[DynamicRankBuffer],):
    assert_param_bool_msg[axis == 0, "_concat_inner only supports axis 0"]()
    var num_elems_copied: Int = 0
    for i in range(inputs.__len__()):
        let input_buf = inputs[i].to_buffer[type]()
        let buffer_len = input_buf.__len__()
        let output_buffer_offset = Buffer[unknown, type](
            output.data.offset(num_elems_copied), buffer_len
        )
        memcpy[type](output_buffer_offset, input_buf)
        num_elems_copied += buffer_len


fn concat[
    type: DType
](
    output: Buffer[unknown, type],
    axis: Int,
    inputs: VariadicList[DynamicRankBuffer],
):
    let input0_dims = inputs[0].shape
    let input0_rank = inputs[0].rank
    # check inputs have same rank and same dims except for axis dim
    for i in range(inputs.__len__()):
        debug_assert(
            input0_rank == inputs[i].rank,
            "all concat inputs must have the same rank",
        )
        for j in range(inputs[i].rank):
            debug_assert(
                Int(input0_dims.load(j)[0].value)
                == Int(inputs[i].shape.load(j)[0].value),
                (
                    "all concat inputs must have the same dimensions in the"
                    " non-concat axes"
                ),
            )

    if axis == 0:
        _concat_inner[type, 0](output, inputs)
        return

    _concat[type](output, axis, inputs)
