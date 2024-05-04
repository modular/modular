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
from gpu import lane_id, WARP_SIZE


fn num_matrix_reg[dim_1: Int, dim_2: Int]() -> Int:
    return (dim_1 * dim_2) // WARP_SIZE


# shapes
alias shape_null = StaticIntTuple[3](0, 0, 0)
alias shape_16x8x4 = StaticIntTuple[3](16, 8, 4)
alias shape_16x8x8 = StaticIntTuple[3](16, 8, 8)
alias shape_16x8x16 = StaticIntTuple[3](16, 8, 16)
alias shape_8x8x4 = StaticIntTuple[3](8, 8, 4)


struct TensorCore[out_type: DType, in_type: DType, shape: StaticIntTuple[3]]:

    """
    Layout reference => https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp#L44.
    """

    # mma tile layouts
    alias tile_null = Layout(IntTuple(0, 0), IntTuple(0, 0))
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
    alias tile_8x8_row = Layout(
        IntTuple(IntTuple(4, 8), 2), IntTuple(IntTuple(16, 1), 8)
    )
    alias tile_8x16_row = Layout(
        IntTuple(IntTuple(4, 8), 4),
        IntTuple(IntTuple(32, 1), 8),
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
            return List[StaticIntTuple[3]](shape_16x8x4, shape_16x8x8)
        elif out_type == DType.float32 and in_type == DType.bfloat16:
            return List[StaticIntTuple[3]](shape_16x8x8, shape_16x8x16)
        elif out_type == DType.float32 and in_type == DType.float16:
            return List[StaticIntTuple[3]](shape_16x8x8, shape_8x8x4)
        else:
            constrained[False, "No valid shape of mma"]()
            return List[StaticIntTuple[3]](shape_null)

    fn load_a(
        inout self,
        a: LayoutTensor,
    ) -> SIMD[in_type, num_matrix_reg[shape[0], shape[2]]()]:
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var a_reg = SIMD[in_type, num_matrix_reg[shape[0], shape[2]]()]()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_k]()

        alias layout_a = self.tile_16x4 if reg_per_thread == 2 else self.tile_16x8

        var mat_a = a.reshape[layout_a]()
        var group_id = int(lane_id()) >> 2
        var group_lane_id = int(lane_id()) % 4

        @parameter
        if in_type == DType.float32:

            @parameter
            if reg_per_thread == 2:
                a_reg[0] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 0]
                )
                a_reg[1] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 1]
                )
            elif reg_per_thread == 4:
                a_reg[0] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 0, 0]
                )
                a_reg[1] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 1, 0]
                )
                a_reg[2] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 0, 1]
                )
                a_reg[3] = rebind[Scalar[in_type]](
                    mat_a[group_lane_id, group_id, 1, 1]
                )
            else:
                constrained[
                    False, "No valid mma shape to load matrix fragment a"
                ]()
        else:
            constrained[False, "No valid type to load matrix fragment a"]()
        return a_reg

    fn load_b(
        inout self,
        b: LayoutTensor,
    ) -> SIMD[in_type, num_matrix_reg[shape[2], shape[1]]()]:
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var b_reg = SIMD[in_type, num_matrix_reg[shape[2], shape[1]]()]()
        alias reg_per_thread = num_matrix_reg[mma_k, mma_n]()

        alias layout_b = self.tile_8x4 if reg_per_thread == 1 else self.tile_8x8

        var mat_b = b.transpose().reshape[layout_b]()
        var group_id = int(lane_id()) >> 2
        var group_lane_id = int(lane_id()) % 4

        @parameter
        if in_type == DType.float32:

            @parameter
            if reg_per_thread == 1:
                b_reg[0] = rebind[Scalar[in_type]](
                    mat_b[group_lane_id, group_id]
                )
            elif reg_per_thread == 2:
                b_reg[0] = rebind[Scalar[in_type]](
                    mat_b[group_lane_id, group_id, 0]
                )
                b_reg[1] = rebind[Scalar[in_type]](
                    mat_b[group_lane_id, group_id, 1]
                )
            else:
                constrained[
                    False, "No valid mma shape to load matrix fragment b"
                ]()
        else:
            constrained[False, "No valid type to load matrix fragment b"]()
        return b_reg

    fn load_c(
        inout self,
        c: LayoutTensor,
    ) -> SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]:
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var c_reg = SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        alias layout_c = self.tile_16x8_row if reg_per_thread == 4 else self.tile_null

        var mat_c = c.reshape[layout_c]()
        var group_id = int(lane_id()) >> 2
        var group_lane_id = int(lane_id()) % 4

        @parameter
        if out_type == DType.float32:

            @parameter
            if reg_per_thread == 4:
                c_reg[0] = rebind[Scalar[out_type]](
                    mat_c[group_id, group_lane_id, 0, 0]
                )
                c_reg[1] = rebind[Scalar[out_type]](
                    mat_c[group_id, group_lane_id, 1, 0]
                )
                c_reg[2] = rebind[Scalar[out_type]](
                    mat_c[group_id, group_lane_id, 0, 1]
                )
                c_reg[3] = rebind[Scalar[out_type]](
                    mat_c[group_id, group_lane_id, 1, 1]
                )
            else:
                constrained[False, "No valid shape to load matrix fragment c"]()
        else:
            constrained[False, "No valid type to load matrix fragment c"]()
        return c_reg

    fn store_d[
        layout_mat: Layout
    ](
        inout self,
        d: LayoutTensor[out_type, layout_mat],
        d_reg: SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()],
    ):
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        alias layout_d = self.tile_16x8_row if reg_per_thread == 4 else self.tile_null

        var mat_d = d.reshape[layout_d]()
        var group_id = int(lane_id()) >> 2
        var group_lane_id = int(lane_id()) % 4

        @parameter
        if out_type == DType.float32:

            @parameter
            if reg_per_thread == 4:
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
                constrained[
                    False, "No valid shape to store to LayoutTensor d"
                ]()
        else:
            constrained[False, "No valid type to store to LayoutTensor d"]()

    fn mma(
        inout self, inout a: SIMD, inout b: SIMD, inout c: SIMD
    ) -> SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]:
        var d = SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]()
        mma(d, a, b, c)
        return d
