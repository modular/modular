# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for using Tensor Cores do to arithmetic and matrix operations
"""

from math import align_down
from sys import simdwidthof, sizeof

from gpu import WARP_SIZE, BlockIdx, ThreadIdx, lane_id
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from layout._utils import load_to_simd
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import LayoutTensor, _swizzle_signature
from layout.swizzle import *

from utils import IndexList


fn num_matrix_reg[dim_1: Int, dim_2: Int]() -> Int:
    return (dim_1 * dim_2) // WARP_SIZE


# shapes
alias shape_null = IndexList[3](0, 0, 0)
alias shape_16x8x4 = IndexList[3](16, 8, 4)
alias shape_16x8x8 = IndexList[3](16, 8, 8)
alias shape_16x8x16 = IndexList[3](16, 8, 16)
alias shape_8x8x4 = IndexList[3](8, 8, 4)


struct TensorCore[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
]:

    """
    Layout reference => https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp#L44.
    """

    alias supported_fp32 = in_type == DType.float32 and shape == shape_16x8x8
    alias supported_half = in_type.is_half_float() and shape == shape_16x8x16

    # Operand register types.
    alias a_reg_type = SIMD[in_type, num_matrix_reg[shape[0], shape[2]]()]
    alias a_reg_tile_type = LayoutTensor[
        in_type,
        Layout.col_major(1, Self.a_reg_type.size),
        address_space = AddressSpace.LOCAL,
    ]
    alias b_reg_type = SIMD[in_type, num_matrix_reg[shape[2], shape[1]]()]
    alias b_reg_tile_type = LayoutTensor[
        in_type,
        Layout.row_major(Self.b_reg_type.size, 1),
        address_space = AddressSpace.LOCAL,
    ]
    alias c_reg_type = SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]
    alias c_reg_tile_type = LayoutTensor[
        out_type,
        Layout.col_major(1, Self.c_reg_type.size),
        address_space = AddressSpace.LOCAL,
    ]

    fn __init__(inout self):
        pass

    @staticmethod
    fn get_shapes[out_type: DType, in_type: DType]() -> List[IndexList[3]]:
        @parameter
        if out_type is DType.float32 and in_type is DType.float32:
            return List[IndexList[3]](shape_16x8x4, shape_16x8x8)
        elif out_type is DType.float32 and in_type is DType.bfloat16:
            return List[IndexList[3]](shape_16x8x8, shape_16x8x16)
        elif out_type is DType.float32 and in_type is DType.float16:
            return List[IndexList[3]](shape_16x8x8, shape_8x8x4)
        else:
            constrained[False, "No valid shape of mma"]()
            return List[IndexList[3]](shape_null)

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn load_a(
        inout self,
        a: LayoutTensor,
    ) -> Self.a_reg_tile_type as res:
        alias mma_m = shape[0]
        alias mma_k = shape[2]
        var a_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_k]()

        alias warp_layout = Layout.row_major(8, 4)

        constrained[
            in_type in (DType.float32, DType.bfloat16, DType.float16),
            "No valid type to load matrix fragment a",
        ]()

        @parameter
        if in_type is DType.float32:
            constrained[
                reg_per_thread in (2, 4),
                "No valid mma shape to load matrix fragment a (float32)",
            ]()
            var a_reg_frags = a.distribute[warp_layout](lane_id())
            a_reg_tile.copy_from(a_reg_frags)

        elif in_type is DType.bfloat16 or in_type is DType.float16:
            constrained[
                reg_per_thread in (4, 8),
                "No valid mma shape to load matrix fragment a (half-float)",
            ]()
            var a_reg_frags = a.vectorize[1, 2]().distribute[warp_layout](
                lane_id()
            )
            a_reg_tile.vectorize[1, 2]().copy_from(a_reg_frags)
        return a_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn load_b(
        inout self,
        b: LayoutTensor,
    ) -> Self.b_reg_tile_type as res:
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var b_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_k, mma_n]()

        alias warp_layout = Layout.row_major(
            8, 4
        ) if transpose_b else Layout.col_major(4, 8)

        @parameter
        if in_type is DType.float32:
            constrained[
                reg_per_thread in (1, 2, 4),
                "No valid mma shape to load matrix fragment b",
            ]()

            var b_ram_frags = b.distribute[warp_layout](lane_id())
            b_reg_tile.copy_from(b_ram_frags)

        elif in_type is DType.bfloat16 or in_type is DType.float16:
            constrained[
                reg_per_thread in (2, 4),
                "No valid mma shape to load matrix fragment b",
            ]()

            var b_ram_frags = b.vectorize[2, 1]().distribute[warp_layout](
                lane_id()
            )
            b_reg_tile.vectorize[2, 1]().copy_from(b_ram_frags)

        else:
            constrained[False, "No valid type to load matrix fragment b"]()
        return b_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn load_c(
        inout self,
        c: LayoutTensor,
    ) -> Self.c_reg_tile_type as res:
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var c_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        @parameter
        if out_type is DType.float32:
            constrained[
                reg_per_thread == 4, "No valid shape to load matrix fragment c"
            ]()

            var c_ram_frags = c.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            c_reg_tile.vectorize[1, 2]().copy_from(c_ram_frags)

        else:
            constrained[False, "No valid type to load matrix fragment c"]()
        return c_reg_tile

    @always_inline
    fn store_d(self, d_dst: LayoutTensor, d_src: LayoutTensor):
        constrained[
            d_dst.dtype == out_type,
            "destination tensor must have the same type",
        ]()
        constrained[
            d_src.shape[0]() == Self.c_reg_tile_type.shape[0]()
            and d_src.shape[1]() == Self.c_reg_tile_type.shape[1](),
            "src tensor must have the same shape as c_reg_tile_type",
        ]()
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()

        @parameter
        if out_type is DType.float32:
            constrained[
                reg_per_thread == 4, "No valid shape to store to LayoutTensor d"
            ]()

            d_dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
                lane_id()
            ).copy_from(d_src.vectorize[1, 2]())

        else:
            constrained[False, "No valid type to store to LayoutTensor d"]()

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn mma_op(
        self,
        a: LayoutTensor,
        b: LayoutTensor,
        c: LayoutTensor,
    ) -> Self.c_reg_tile_type as res:
        var a_reg = load_to_simd(a)
        var b_reg = load_to_simd(b)
        var c_reg = load_to_simd(c)
        var d_reg = Self.c_reg_type()
        mma(d_reg, a_reg, b_reg, c_reg)
        var d = __type_of(res).stack_allocation()
        d.vectorize[1, Self.c_reg_type.size]()[0, 0] = rebind[
            __type_of(d.vectorize[1, Self.c_reg_type.size]()[0, 0])
        ](d_reg)
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
                    alias num_frags_round_even = align_down(num_frags, 2)

                    @parameter
                    for i in range(0, num_frags_round_even, 2):
                        # load using x4 layout
                        var vec = _load_matrix_frag[transposed=True](
                            mma_tile, i
                        )

                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])

                    @parameter
                    if num_frags % 2:
                        # load using x2 for the last fragment if necessary
                        var vec = _load_matrix_frag[
                            transposed=True, num_matrices=2
                        ](mma_tile, num_frags_round_even)
                        fragments[num_frags_round_even, 0] = rebind[frag_type](
                            vec
                        )

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
    # swizzle: OptionalReg[_swizzle_signature] = None,
    swizzle: Bool = True,
    transposed: Bool = False,
    x4_row_major: Bool = False,
    num_matrices: Int = 4,
    *,
    # Work around parameter deduction MOCO-854.
    __type: DType,
    __layout: Layout,
    # Nvidia GPU register is 4B.
    __register_width: Int = 4,
    __output_width: Int = num_matrices * __register_width // sizeof[__type](),
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
    #
    # This function will also work if num_matrices is 1 or 2, in that case
    # ld_matrix will call ldmatrix with num = x1 or x2, num depends
    # on __output_width which in turn depends on num_matrices.
    # lane_offset based on x4 will also work because in case of x1 and x2
    # ld_matrix ignores pointers for lane >= 8 and lane >= 16 respectively.
    alias ldmatrix_threadmap = Layout.col_major(16, 2)

    # 4 submatrices layout
    alias x4_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(num_mat_per_row, 1, 8 * num_mat_per_row)
    ) if x4_row_major else Layout(IntTuple(16, 2), IntTuple(num_mat_per_row, 1))

    alias ldmatrix_layout = ComposedLayout(
        x4_layout,
        make_ldmatrix_swizzle[
            mma_tile.dtype, row_size
        ]() if swizzle else Swizzle(0, 0, 1),
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
]() -> IndexList[3]:
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
fn get_fragment_size[mma_shape: IndexList[3]]() -> IndexList[3]:
    return IndexList[3](
        mma_shape[0] * mma_shape[2] // WARP_SIZE,
        mma_shape[1] * mma_shape[2] // WARP_SIZE,
        mma_shape[0] * mma_shape[1] // WARP_SIZE,
    )
