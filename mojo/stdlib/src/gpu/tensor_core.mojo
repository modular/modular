# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Tensor Cores do to arithmetic and matrix operations
"""

from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor
from gpu.mma import mma


fn compare_shape[s1: StaticIntTuple[3], s2: StaticIntTuple[3]]() -> Bool:
    @parameter
    if s1[0] == s2[0] and s1[1] == s2[1] and s1[2] == s2[2]:
        return True
    else:
        return False


struct TensorCore:
    """
    Layout reference => https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp#L44.
    """

    # for fp32_fp32
    alias tile_16x4 = Layout(
        IntTuple(IntTuple(4, 8), 2), IntTuple(IntTuple(16, 1), 8)
    )
    alias tile_8x4 = Layout(
        IntTuple(IntTuple(4, 8), 1), IntTuple(IntTuple(8, 1), 0)
    )
    alias tile_16x8 = Layout(
        IntTuple(IntTuple(4, 8), IntTuple(2, 2)),
        IntTuple(IntTuple(16, 1), IntTuple(8, 64)),
    )
    alias tile_8x8 = Layout(
        IntTuple(IntTuple(4, 8), 2), IntTuple(IntTuple(8, 1), 32)
    )
    alias tile_16x8_row = Layout(
        IntTuple(IntTuple(4, 8), IntTuple(2, 2)),
        IntTuple(IntTuple(32, 1), IntTuple(16, 8)),
    )

    fn __init__(inout self):
        pass

    @staticmethod
    fn get_shapes[out_type: DType, in_type: DType]() -> List[StaticIntTuple[3]]:
        @parameter
        if out_type == DType.float32 and in_type == DType.float32:
            return List[StaticIntTuple[3]](
                StaticIntTuple[3](16, 8, 4), StaticIntTuple[3](16, 8, 8)
            )
        else:
            constrained[False, "No valid shape of mma"]()
            return List[StaticIntTuple[3]]((0, 0, 0))

    fn mma[
        out_type: DType, in_type: DType, shape: StaticIntTuple[3]
    ](
        inout self,
        d: LayoutTensor,
        a: LayoutTensor,
        b: LayoutTensor,
        c: LayoutTensor,
    ):
        @parameter
        if out_type == DType.float32 and in_type == DType.float32:

            @parameter
            if compare_shape[shape, StaticIntTuple[3](16, 8, 4)]():
                pass
            elif compare_shape[shape, StaticIntTuple[3](16, 8, 8)]():
                alias layout_a = self.tile_16x8
                alias layout_b = self.tile_8x8
                alias layout_c = self.tile_16x8_row

                var mat_a = a.reshape[layout_a]()
                var mat_b = b.transpose().reshape[layout_b]()
                var mat_c = c.reshape[layout_c]()
                var mat_d = d.reshape[layout_c]()

                var group_id = int(lane_id()) >> 2
                var group_lane_id = int(lane_id()) % 4

                var a_reg = SIMD[DType.float32, 4](
                    rebind[Float32](mat_a[group_lane_id, group_id, 0, 0]),
                    rebind[Float32](mat_a[group_lane_id, group_id, 1, 0]),
                    rebind[Float32](mat_a[group_lane_id, group_id, 0, 1]),
                    rebind[Float32](mat_a[group_lane_id, group_id, 1, 1]),
                )
                var b_reg = SIMD[DType.float32, 2](
                    rebind[Float32](mat_b[group_lane_id, group_id, 0]),
                    rebind[Float32](mat_b[group_lane_id, group_id, 1]),
                )
                var c_reg = SIMD[DType.float32, 4](
                    rebind[Float32](mat_c[group_id, group_lane_id, 0, 0]),
                    rebind[Float32](mat_c[group_id, group_lane_id, 1, 0]),
                    rebind[Float32](mat_c[group_id, group_lane_id, 0, 1]),
                    rebind[Float32](mat_c[group_id, group_lane_id, 1, 1]),
                )

                var d_reg = SIMD[DType.float32, 4](0)

                mma(d_reg, a_reg, b_reg, c_reg)

                mat_d[group_lane_id, group_id, 0, 0] = rebind[
                    mat_d.element_type
                ](d_reg[0])
                mat_d[group_lane_id, group_id, 1, 0] = rebind[
                    mat_d.element_type
                ](d_reg[1])
                mat_d[group_lane_id, group_id, 0, 1] = rebind[
                    mat_d.element_type
                ](d_reg[2])
                mat_d[group_lane_id, group_id, 1, 1] = rebind[
                    mat_d.element_type
                ](d_reg[3])
            else:
                constrained[False, "no valid shape of mma types (FP32, FP32)"]()
