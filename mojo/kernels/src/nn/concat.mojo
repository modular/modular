# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, DynamicRankBuffer
from List import VariadicList
from Memory import memcpy
from Range import range
from Int import Int

# ===----------------------------------------------------------------------===#
# concat
# ===----------------------------------------------------------------------===#


fn concat[
    type: __mlir_type.`!kgen.dtype`
](
    output: Buffer[__mlir_attr.`#kgen.unknown : index`, type],
    axis: Int,
    inputs: VariadicList[DynamicRankBuffer],
):
    let buffer_len = inputs[0].to_buffer[type]().__len__()
    for i in range(inputs.__len__()):
        let output_buffer_offset = Buffer[
            __mlir_attr.`#kgen.unknown : index`, type
        ](output.data.offset(i * buffer_len), buffer_len)
        memcpy[type](output_buffer_offset, inputs[i].to_buffer[type]())
