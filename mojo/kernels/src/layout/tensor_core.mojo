# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Tensor Cores do to arithmetic and matrix operations
"""

from sys import simdwidthof, sizeof

from gpu import WARP_SIZE, BlockIdx, ThreadIdx, lane_id
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor, _swizzle_signature
from layout.swizzle import *

from utils import StaticIntTuple


fn num_matrix_reg[dim_1: Int, dim_2: Int]() -> Int:
    return (dim_1 * dim_2) // WARP_SIZE


# shapes
alias shape_null = StaticIntTuple[3](0, 0, 0)
alias shape_16x8x4 = StaticIntTuple[3](16, 8, 4)
alias shape_16x8x8 = StaticIntTuple[3](16, 8, 8)
alias shape_16x8x16 = StaticIntTuple[3](16, 8, 16)
alias shape_8x8x4 = StaticIntTuple[3](8, 8, 4)


struct TensorCore[
    out_type: DType,
    in_type: DType,
    shape: StaticIntTuple[3],
    transpose_b: Bool = False,
]:

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
    alias tile_16x16_row = Layout(
        IntTuple(IntTuple(4, 8), IntTuple(2, 2, 2)),
        IntTuple(IntTuple(31, 1), IntTuple(16, 8, 128)),
    )

    alias supported_fp32 = in_type == DType.float32 and shape == shape_16x8x8
    alias supported_half = in_type.is_half_float() and shape == shape_16x8x16

    fn __init__(inout self):
        pass

    @staticmethod
    fn get_shapes[out_type: DType, in_type: DType]() -> List[StaticIntTuple[3]]:
        @parameter
        if out_type is DType.float32 and in_type is DType.float32:
            return List[StaticIntTuple[3]](shape_16x8x4, shape_16x8x8)
        elif out_type is DType.float32 and in_type is DType.bfloat16:
            return List[StaticIntTuple[3]](shape_16x8x8, shape_16x8x16)
        elif out_type is DType.float32 and in_type is DType.float16:
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

        alias layout_tf32 = self.tile_16x4 if reg_per_thread == 2 else self.tile_16x8
        alias layout_f16 = self.tile_16x8_row if reg_per_thread == 4 else self.tile_16x16_row
        alias layout_a = (layout_f16) if (
            in_type is DType.bfloat16 or in_type is DType.float16
        ) else (layout_tf32)

        var mat_a = a.composition[layout_a]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        @parameter
        if in_type is DType.float32:

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
        elif in_type is DType.bfloat16 or in_type is DType.float16:

            @parameter
            if reg_per_thread == 4:
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

        alias layout_tf32 = self.tile_8x4 if reg_per_thread == 1 else self.tile_8x8
        alias layout_f16 = self.tile_8x8_row if reg_per_thread == 2 else self.tile_16x8
        alias layout_b = (layout_f16) if (
            in_type is DType.bfloat16 or in_type is DType.float16
        ) else (layout_tf32)

        var mat_b = b.transpose().composition[layout_b]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        @parameter
        if in_type is DType.float32:

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
        elif in_type is DType.bfloat16 or in_type is DType.float16:

            @parameter
            if reg_per_thread == 2:
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

        var mat_c = c.composition[layout_c]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        @parameter
        if out_type is DType.float32:

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

        var mat_d = d.composition[layout_d]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        @parameter
        if out_type is DType.float32:

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

    @always_inline
    fn load_a[
        swizzle: Bool = True,
        *,
        type0: DType,
        layout0: Layout,
        element_layout0: Layout,
        type1: DType,
        layout1: Layout,
        element_layout1: Layout,
    ](
        self,
        warp_tile: LayoutTensor[
            type0,
            layout0,
            address_space = AddressSpace.SHARED,
            element_layout=element_layout0,
        ],
        fragments: LayoutTensor[
            type1,
            layout1,
            element_layout=element_layout1,
            address_space = AddressSpace.LOCAL,
        ],
        mma_tile_coordk: Int = 0,  # the k corrdinate of mma tile
    ):
        constrained[self.supported_fp32 or self.supported_half]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[type0]()
        alias num_frags = fragments.shape[0]()

        var swizzle_offset = mma_tile_coordk * shape[2] // simd_size

        @parameter
        for i in range(num_frags):
            var mma_tile = warp_tile.tile[shape[0], warp_tile.shape[1]()](i, 0)
            fragments[i, 0] = rebind[frag_type](
                _load_matrix_frag[swizzle](mma_tile, swizzle_offset)
            )

    @always_inline
    fn load_b[
        *,
        type0: DType,
        layout0: Layout,
        element_layout0: Layout,
        layout1: Layout,
        element_layout1: Layout,
    ](
        self,
        warp_tile: LayoutTensor[
            type0,
            layout0,
            address_space = AddressSpace.SHARED,
            element_layout=element_layout0,
        ],
        fragments: LayoutTensor[
            type0,
            layout1,
            element_layout=element_layout1,
            address_space = AddressSpace.LOCAL,
        ],
        mma_tile_coordk: Int = 0,  # the k corrdinate of mma tile
        warp_tile_coordn: Int = 0,  # n coordiante of warp tile
    ):
        constrained[self.supported_fp32 or self.supported_half]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[in_type]()
        alias num_frags = fragments.shape[0]()
        alias WN = warp_tile.shape[1]()

        @parameter
        if transpose_b:

            @parameter
            if in_type == DType.float32:
                var swizzle_offset = mma_tile_coordk * shape[2] // simd_size

                @parameter
                for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag(mma_tile, swizzle_offset)
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[type0, 2](vec[0], vec[2])
                    )
                    fragments[i + 1, 0] = rebind[frag_type](
                        SIMD[type0, 2](vec[1], vec[3])
                    )
            else:
                constrained[self.supported_half]()

                var swizzle_offset = mma_tile_coordk * shape[2] // simd_size

                @parameter
                for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag[x4_row_major=True](
                        mma_tile, swizzle_offset
                    )
                    var high_low = vec.split()
                    fragments[i, 0] = rebind[frag_type](high_low[0])
                    fragments[i + 1, 0] = rebind[frag_type](high_low[1])

        else:

            @parameter
            if in_type == DType.float32:

                @parameter
                for i in range(num_frags):
                    var mma_tile = warp_tile.tile[shape[2], shape[1]](
                        mma_tile_coordk, i
                    )
                    var frag = mma_tile.distribute[Layout.col_major(4, 8)](
                        lane_id()
                    )
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[type0, 2](
                            rebind[Scalar[type0]](frag[0]),
                            rebind[Scalar[type0]](frag[1]),
                        )
                    )

            else:
                constrained[self.supported_half]()

                var mma_tile = warp_tile.tile[shape[2], warp_tile.shape[1]()](
                    mma_tile_coordk, 0
                )

                # This is a hack to get correct result for small warp tile.
                # If we swizzle 3 bits, 8 simd vectors repeats a pattern,
                # and if WN = 32 = 4 simd vectors, the result would be wrong
                # because 2nd warp tile doesn't know it's in the middle of a pattern.
                # The hack shifts back the pointer and use idx in shared memory tile
                # to do the right swizzling.
                # The potential fix is to have both base pointer and offset inside
                # Layout tensor so the warp_tile has the original address of the
                # shared memory tile.
                @parameter
                if WN == 32:  # 32 is the min in practice.
                    var mma_tile_shifted = LayoutTensor[
                        mma_tile.dtype,
                        mma_tile.layout,
                        address_space = AddressSpace.SHARED,
                    ](mma_tile.ptr - warp_tile_coordn * WN)

                    @parameter
                    for i in range(0, num_frags, 2):
                        var swizzle_offset = (
                            i + warp_tile_coordn * WN // simd_size
                        )
                        var vec = _load_matrix_frag[transposed=True](
                            mma_tile_shifted, swizzle_offset
                        )
                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])
                else:

                    @parameter
                    for i in range(0, num_frags, 2):
                        var vec = _load_matrix_frag[transposed=True](
                            mma_tile, i
                        )
                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])

    @always_inline
    fn mma(
        self, a_frag: LayoutTensor, b_frag: LayoutTensor, c_frag: LayoutTensor
    ):
        # TODO: Assume that fragments are all vectorized layout tensor with
        # dims num_vectors x 1. Consider using TensorCore to allocate fragments
        # so the caller don't explicitly maintain the shape.
        alias num_m_mmas = a_frag.shape[0]()
        alias num_n_mmas = b_frag.shape[0]()

        constrained[
            c_frag.shape[0]() == num_m_mmas * num_n_mmas,
            "Fragments size mismatch.",
        ]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                mma(
                    c_frag[n_mma * num_m_mmas + m_mma, 0],
                    a_frag[m_mma, 0],
                    b_frag[n_mma, 0],
                    c_frag[n_mma * num_m_mmas + m_mma, 0],
                )


@always_inline
fn _load_matrix_frag[
    # Refactor the three parameters with ComposedLayout
    # swizzle: Optional[_swizzle_signature] = None,
    swizzle: Bool = True,
    transposed: Bool = False,
    x4_row_major: Bool = False,
    *,
    # Work around parameter deduction MOCO-854.
    __type: DType,
    __layout: Layout,
    # Nvidia GPU register is 4B.
    __register_width: Int = 4,
    __num_matrices: Int = 4,
    __output_width: Int = __num_matrices * __register_width // sizeof[__type](),
](
    mma_tile: LayoutTensor[
        __type,
        __layout,
        address_space = AddressSpace.SHARED,
    ],
    offset: Int,
) -> SIMD[mma_tile.dtype, __output_width]:
    alias simd_size = simdwidthof[mma_tile.dtype]()

    # mma_tile is tiled from the row major shared memory buffer. Retrieve the
    # buffer's stride for computing the swizzle.
    alias row_size = mma_tile.stride[0]()
    alias num_mat_per_row = row_size // simd_size

    var lane = lane_id()

    # We load 4 matrices a time for max throughput. Each matrix has 8 vectors
    # and each thread loads one vector. The 4 matrices for 16x8x8 and 16x8x16
    # could be arranged in column or row-major.
    #
    #         |--------|--------|            |--------|--------|
    #         | mat 0  | mat 2  |            | mat 0  | mat 1  |
    #         |--------|--------|            |--------|--------|
    #         | mat 1  | mat 3  |            | mat 2  | mat 3  |
    #         |--------|--------|            |--------|--------|
    #            A 16x16  or                 B Transposed 2 16x8
    #            B 2x 16x8
    #
    # Left is for A since it match A's mma tile layout exactly. It's also for B
    # 16x8x16 when two 16x8 matrices are grouped in one load (using ldmatrix.trans).
    # When B is *transposed*, we arrage 4 matrices in row-major so that mat0-1
    # contribute to one mma's fragment.
    # !!! Don't use column major and pass mat0, mat2's register to HMMA. This
    # hits undocumented register conflicts and is very slow !!!

    # We load 4 matrices a time for max throughput. Each matrix has 8 vectors
    # and each thread loads one vector. For mma shape 16x8 or 16x16, the 4
    # matrices are arranged in column major.
    alias ldmatrix_threadmap = Layout.col_major(16, 2)

    # 4 submatrices layout
    alias x4_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(num_mat_per_row, 1, 8 * num_mat_per_row)
    ) if x4_row_major else Layout(IntTuple(16, 2), IntTuple(num_mat_per_row, 1))

    alias ldmatrix_layout = ComposedLayout(
        x4_layout,
        make_ldmatrix_swizzleex[
            mma_tile.dtype, row_size
        ]() if swizzle else SwizzleEx(0, 0, 1),
    )

    var lane_offset = eval_composed[ldmatrix_layout](
        int(lane), offset
    ) * simd_size

    return ld_matrix[mma_tile.dtype, __output_width, transposed](
        mma_tile.ptr + lane_offset
    )


@always_inline
fn get_mma_shape[
    input_type: DType, accum_type: DType, shape_id: Int = 0
]() -> StaticIntTuple[3]:
    @parameter
    if accum_type is DType.float32 and input_type is DType.float32:

        @parameter
        if shape_id == 0:
            return shape_16x8x8
        else:
            return shape_16x8x4

    elif accum_type is DType.float32 and input_type is DType.bfloat16:

        @parameter
        if shape_id == 0:
            return shape_16x8x16
        else:
            return shape_16x8x8

    elif accum_type is DType.float32 and input_type is DType.float16:

        @parameter
        if shape_id == 0:
            return shape_16x8x16
        elif shape_id == 1:
            return shape_16x8x8
        else:
            return shape_8x8x4
    else:
        constrained[False, "Unsupported mma shape."]()
        return shape_null


@always_inline
fn get_accum_type[
    input_type: DType, preferred_accum_type: DType = input_type
]() -> DType:
    @parameter
    if input_type is DType.float32:
        return DType.float32
    elif input_type is DType.bfloat16:
        return DType.float32
    # fp16 accumulation can be done in fp16 or fp32. Use fp16 by default for better
    # performance and use fp32 only when it's specified via preferred type.
    elif input_type is DType.float16:

        @parameter
        if preferred_accum_type is DType.float32:
            return preferred_accum_type
        else:
            return DType.float16
    else:
        constrained[
            False, "Only support fp16, bf16, fp32 accumulation for now."
        ]()
        return input_type


@always_inline
fn get_fragment_size[mma_shape: StaticIntTuple[3]]() -> StaticIntTuple[3]:
    return StaticIntTuple[3](
        mma_shape[0] * mma_shape[2] // WARP_SIZE,
        mma_shape[1] * mma_shape[2] // WARP_SIZE,
        mma_shape[0] * mma_shape[1] // WARP_SIZE,
    )
