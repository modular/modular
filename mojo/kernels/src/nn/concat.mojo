# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, DynamicRankBuffer
from DType import DType
from Int import Int
from List import VariadicList
from Memory import memcpy
from Range import range

# ===----------------------------------------------------------------------===#
# concat
# ===----------------------------------------------------------------------===#


fn concat[
    type: DType
](
    output: Buffer[__mlir_attr.`#kgen.unknown : index`, type],
    axis: Int,
    inputs: VariadicList[DynamicRankBuffer],
):
    var num_elems_copied: Int = 0
    for i in range(inputs.__len__()):
        let input_buf = inputs[i].to_buffer[type]()
        let buffer_len = input_buf.__len__()
        let output_buffer_offset = Buffer[
            __mlir_attr.`#kgen.unknown : index`, type
        ](output.data.offset(num_elems_copied), buffer_len)
        memcpy[type](output_buffer_offset, input_buf)
        num_elems_copied += buffer_len
