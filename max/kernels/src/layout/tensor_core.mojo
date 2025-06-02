# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
Tensor Core Module for High-Performance Matrix Operations

Provides abstractions for using GPU Tensor Cores to perform optimized matrix operations.
It supports both NVIDIA and AMD GPU architectures with hardware-specific optimizations.

Key Components:
--------------
- `TensorCore`: Core struct that encapsulates tensor core operations with support for various
  data types and matrix shapes. It handles loading matrix fragments, performing matrix
  multiply-accumulate operations, and storing results.

- Matrix Fragment Management: Functions for loading and storing matrix fragments to/from
  shared memory with hardware-specific optimizations.

- Matrix Multiply-Accumulate (MMA): Optimized implementations of matrix multiplication
  operations using tensor cores.

Supported Operations:
-------------------
- Matrix loading with various layouts and swizzling patterns
- Matrix multiply-accumulate (D = A * B + C)
- Matrix storing with hardware-specific optimizations

Supported Data Types:
-------------------
- NVIDIA: float32, bfloat16, float16, float8_e4m3fn, float8_e5m2
- AMD: float32, bfloat16, float16

Supported Matrix Shapes:
----------------------
- NVIDIA: 16×8×8, 16×8×4, 16×8×16, 8×8×4, 16×8×32
- AMD: 16×16×4, 16×16×16, 32×32×8
"""

from collections import OptionalReg
from math import align_down
from sys import (
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simdwidthof,
    sizeof,
)

from gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from gpu.intrinsics import lop
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from layout._utils import load_to_simd
from layout.int_tuple import IntTuple, product
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.swizzle import (
    ComposedLayout,
    Swizzle,
    eval_composed,
    make_ldmatrix_swizzle,
)
from memory.unsafe import bitcast
from stdlib.builtin.simd import _has_native_f8_support

from utils import IndexList
from utils.index import Index


fn num_matrix_reg[dim_1: Int, dim_2: Int]() -> Int:
    """Calculates the number of matrix registers required per thread.

    Determines how many registers each thread in a warp needs to store a matrix
    of the given dimensions. This is calculated by dividing the total number of
    elements (dim_1 * dim_2) by the warp size, as the matrix is distributed
    across all threads in the warp.

    Parameters:
        dim_1: First dimension of the matrix.
        dim_2: Second dimension of the matrix.

    Returns:
        The number of matrix registers needed per thread.
    """
    return (dim_1 * dim_2) // WARP_SIZE


# shapes
alias shape_null = IndexList[3](0, 0, 0)
alias shape_16x8x4 = IndexList[3](16, 8, 4)
alias shape_16x8x8 = IndexList[3](16, 8, 8)
alias shape_16x8x16 = IndexList[3](16, 8, 16)
alias shape_8x8x4 = IndexList[3](8, 8, 4)
alias shape_16x8x32 = IndexList[3](16, 8, 32)

# MI300x shapes
alias shape_16x16x4 = IndexList[3](16, 16, 4)
alias shape_16x16x16 = IndexList[3](16, 16, 16)
alias shape_32x32x8 = IndexList[3](32, 32, 8)


fn _get_a_k_group_size[a: Layout, shape: IndexList[3]]() -> Int:
    return product(a.shape[1]) // shape[2]


fn _get_b_k_group_size[
    b: Layout, shape: IndexList[3], transpose_b: Bool
]() -> Int:
    return (
        product(b.shape[1])
        // shape[2] if transpose_b else product(b.shape[0])
        // shape[2]
    )


fn _get_a_reg_tile_layout[a: Layout, shape: IndexList[3]]() -> Layout:
    return Layout.col_major(
        1,
        num_matrix_reg[shape[0], shape[2]]() * _get_a_k_group_size[a, shape](),
    )


fn _get_b_reg_tile_layout[
    b: Layout, shape: IndexList[3], transpose_b: Bool
]() -> Layout:
    return Layout.row_major(
        num_matrix_reg[shape[2], shape[1]]()
        * _get_b_k_group_size[b, shape, transpose_b](),
        1,
    )


struct TensorCore[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
]:
    """TensorCore provides an abstraction for GPU tensor core hardware to perform optimized matrix operations.

    This struct encapsulates the functionality required to efficiently map matrix operations to Tensor Cores
    on NVIDIA and AMD GPUs. It handles loading matrix fragments, performing matrix multiply-accumulate
    operations, and storing results with hardware-specific optimizations.

    Parameters:
        out_type: The data type for output/accumulation operations.
        in_type: The data type for input matrix elements.
        shape: The shape parameters for the matrix operation in the form [M, N, K]
               where M×N is the output shape and K is the inner dimension.
        transpose_b: Whether to transpose the B matrix before multiplication. Defaults to False.

    Note:
        Different shapes and data types are supported depending on the GPU hardware.
        For NVIDIA GPUs:
          - float32: 16×8×8 or 16×8×4
          - half-precision: 16×8×16
          - float8: 16×8×32
        For AMD GPUs:
          - float32: 16×16×4
          - half-precision: 16×16×16 or 32×32×8
    """

    # Layout reference => https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp#L44.

    alias supported_fp32 = in_type is DType.float32 and (
        shape == shape_16x8x8 if is_nvidia_gpu() else shape == shape_16x16x4
    )
    alias supported_half = in_type.is_half_float() and (
        shape
        == shape_16x8x16 if is_nvidia_gpu() else shape
        in (shape_16x16x16, shape_32x32x8)
    )
    alias supported_fp8 = in_type in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    ) and shape == shape_16x8x32

    # Operand register types.
    alias a_reg_type = SIMD[in_type, num_matrix_reg[shape[0], shape[2]]()]
    alias b_reg_type = SIMD[in_type, num_matrix_reg[shape[2], shape[1]]()]
    alias c_reg_type = SIMD[out_type, num_matrix_reg[shape[0], shape[1]]()]

    alias c_reg_tile_type = LayoutTensor[
        out_type,
        Layout.col_major(1, Self.c_reg_type.size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    fn __init__(out self):
        """
        Initialize a new TensorCore instance.
        """
        pass

    @staticmethod
    fn get_shapes[out_type: DType, in_type: DType]() -> List[IndexList[3]]:
        """
        Get supported shapes for given data types.

        Returns a list of valid shapes for the specified output and input data types.

        Parameters:
            out_type: The output/accumulation data type.
            in_type: The input matrix data type.

        Returns:
            List[IndexList[3]]: Valid shapes for the matrix operations given the specified types.

        Note:
            The returned shapes are hardware-dependent. Different shapes are supported
            for different combinations of input and output types.
        """

        @parameter
        if out_type is DType.float32 and in_type is DType.float32:
            return List[IndexList[3]](shape_16x8x4, shape_16x8x8)
        elif out_type is DType.float32 and in_type is DType.bfloat16:
            return List[IndexList[3]](shape_16x8x8, shape_16x8x16)
        elif out_type is DType.float32 and in_type is DType.float16:
            return List[IndexList[3]](shape_16x8x8, shape_8x8x4)
        elif out_type is DType.float32 and (
            in_type is DType.float8_e4m3fn or in_type is DType.float8_e5m2
        ):
            return List[IndexList[3]](shape_16x8x32)
        else:
            constrained[False, "No valid shape of mma"]()
            return List[IndexList[3]](shape_null)

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid

    @always_inline
    fn load_a[
        swizzle: OptionalReg[Swizzle] = None
    ](
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_a_reg_tile_layout[a.layout, shape](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        """
        Load the A matrix fragments.

        Loads matrix A from memory into a LayoutTensor suitable for tensor core operations.

        Parameters:
            swizzle: Optional swizzle pattern for optimal memory access (AMD only).

        Args:
            a: The source matrix A data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.
        """

        @parameter
        if is_nvidia_gpu():
            constrained[swizzle is None, "Swizzle is not supported on NVIDIA"]()
            return self._load_a_nvidia(a)
        else:
            return self._load_a_amd[swizzle](a)

    @always_inline
    fn _load_a_amd[
        swizzle: OptionalReg[Swizzle]
    ](
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_a_reg_tile_layout[a.layout, shape](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        alias mma_m = shape[0]
        alias mma_k = shape[2]
        var a_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_k]()
        # for AMD we load k_group_size mma tiles at a time so that we can use 16B loads
        # For example, when loading 16x16 bfloat16 tile only 32 lanes will be active
        # when using 16B loads, so instead we load 16x32 tile in one go.
        alias k_group_size = _get_a_k_group_size[a.layout, shape]()
        alias simd_width = reg_per_thread * k_group_size

        alias warp_layout = Layout.col_major(mma_m, WARP_SIZE // mma_m)

        constrained[
            in_type
            in (
                DType.float32,
                DType.bfloat16,
                DType.float16,
            ),
            "No valid type to load matrix fragment a",
        ]()

        @parameter
        if in_type in (DType.float32, DType.bfloat16, DType.float16):
            constrained[
                (reg_per_thread in (1,) and in_type is DType.float32)
                or (
                    reg_per_thread in (4,)
                    and (in_type in (DType.bfloat16, DType.float16))
                ),
                "No valid mma shape to load matrix fragment",
            ]()
            var a_reg_frags = a.vectorize[1, simd_width]().distribute[
                warp_layout, swizzle=swizzle
            ](lane_id())
            a_reg_tile.vectorize[1, simd_width]().copy_from(a_reg_frags)
        else:
            constrained[
                False,
                "Data type ",
                String(in_type),
                " is not supported for loading matrix A fragments on AMD",
                " GPUs. Only float32, bfloat16 and float16 are supported.",
            ]()
        return a_reg_tile

    @always_inline
    fn _load_a_nvidia(
        self,
        a: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_a_reg_tile_layout[a.layout, shape](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        alias mma_m = shape[0]
        alias mma_k = shape[2]
        var a_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_k]()

        alias warp_layout = Layout.row_major(8, 4)

        constrained[
            in_type
            in (
                DType.float32,
                DType.bfloat16,
                DType.float16,
                DType.float8_e4m3fn,
                DType.float8_e5m2,
            ),
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
        elif in_type is DType.float8_e4m3fn or in_type is DType.float8_e5m2:
            constrained[
                _has_native_f8_support(),
                "float8 formats are only supported in SM90+",
            ]()
            constrained[
                reg_per_thread in (16,),
                "No valid mma shape to load matrix fragment a (half-float)",
            ]()
            var a_reg_frags = a.vectorize[1, 4]().distribute[warp_layout](
                lane_id()
            )
            a_reg_tile.vectorize[1, 4]().copy_from(a_reg_frags)
        return a_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn load_b[
        swizzle: OptionalReg[Swizzle] = None
    ](
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_b_reg_tile_layout[b.layout, shape, transpose_b](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        """
        Load the B matrix fragments.

        Loads matrix B from memory into a `LayoutTensor` suitable for tensor core operations.
        The function handles different hardware architectures and memory access patterns.

        Parameters:
            swizzle: Optional swizzle pattern for optimal memory access (AMD only).
                     Will cause an error if used with NVIDIA GPUs.

        Args:
            b: The source matrix B data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.

        Note:
            If transpose_b is `True`, the B matrix will be transposed during loading.
            This is more efficient than transposing the matrix in memory.
        """

        @parameter
        if is_nvidia_gpu():
            constrained[swizzle is None, "Swizzle is not supported on NVIDIA"]()
            return self._load_b_nvidia(b)
        else:
            return self._load_b_amd[swizzle](b)

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn _load_b_amd[
        swizzle: OptionalReg[Swizzle]
    ](
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_b_reg_tile_layout[b.layout, shape, transpose_b](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var b_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_k, mma_n]()
        alias k_group_size = _get_b_k_group_size[b.layout, shape, transpose_b]()
        alias simd_width = reg_per_thread * k_group_size

        alias warp_layout = Layout.col_major(
            mma_n, WARP_SIZE // mma_n
        ) if transpose_b else Layout.row_major(WARP_SIZE // mma_n, mma_n)

        @parameter
        if in_type in (DType.float32, DType.bfloat16, DType.float16):
            constrained[
                (reg_per_thread in (1,) and in_type is DType.float32)
                or (
                    reg_per_thread in (4,)
                    and (in_type in (DType.bfloat16, DType.float16))
                ),
                "No valid mma shape to load matrix fragment b",
            ]()

            @parameter
            if transpose_b:
                var b_ram_frags = b.vectorize[1, simd_width]().distribute[
                    warp_layout, swizzle=swizzle
                ](lane_id())
                b_reg_tile.vectorize[simd_width, 1]().copy_from(b_ram_frags)
            else:
                var b_ram_frags = b.vectorize[simd_width, 1]().distribute[
                    warp_layout, swizzle=swizzle
                ](lane_id())
                b_reg_tile.vectorize[simd_width, 1]().copy_from(b_ram_frags)
        else:
            constrained[
                False,
                "Data type ",
                String(in_type),
                " is not supported for loading matrix B fragments on AMD",
                " GPUs. Only float32, bfloat16 and float16 are supported.",
            ]()

        return b_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn _load_b_nvidia(
        self,
        b: LayoutTensor,
        out res: LayoutTensor[
            in_type,
            _get_b_reg_tile_layout[b.layout, shape, transpose_b](),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ],
    ):
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

            @parameter
            if transpose_b:
                var b_ram_frags = b.vectorize[1, 2]().distribute[warp_layout](
                    lane_id()
                )
                b_reg_tile.vectorize[2, 1]().copy_from(b_ram_frags.transpose())
            else:
                var b_ram_frags = b.vectorize[2, 1]().distribute[warp_layout](
                    lane_id()
                )
                b_reg_tile.vectorize[2, 1]().copy_from(b_ram_frags)
        elif in_type is DType.float8_e4m3fn or in_type is DType.float8_e5m2:
            constrained[
                reg_per_thread in (8,),
                "No valid mma shape to load matrix fragment b",
            ]()

            var b_ram_frags = b.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            b_reg_tile.vectorize[4, 1]().copy_from(b_ram_frags)

        else:
            constrained[False, "No valid type to load matrix fragment b"]()
        return b_reg_tile

    # need always_inline, otherwise the stack allocated LayoutTensor will not be valid
    @always_inline
    fn load_c(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
        """
        Load the C matrix fragments.

        Loads matrix C from memory into a `LayoutTensor` suitable for tensor core operations.
        The function handles different hardware architectures and memory access patterns.

        Args:
            c: The source matrix C data.

        Returns:
            The loaded matrix fragments as a `LayoutTensor`.
        """

        @parameter
        if is_nvidia_gpu():
            return self._load_c_nvidia(c)
        else:
            return self._load_c_amd(c)

    @always_inline
    fn _load_c_amd(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias mma_k = shape[2]
        var c_reg_tile = __type_of(res).stack_allocation()
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()
        alias warp_layout = Layout.row_major(mma_m // reg_per_thread, mma_n)

        @parameter
        if out_type is DType.float32:
            constrained[
                reg_per_thread in (4, 16),
                "No valid shape to load matrix fragment c",
            ]()

            var c_ram_frags = c.vectorize[4, 1]().distribute[warp_layout](
                lane_id()
            )
            c_reg_tile.vectorize[1, 4]().copy_from(c_ram_frags)
        else:
            constrained[False, "No valid type to load matrix fragment c"]()
        return c_reg_tile

    @always_inline
    fn _load_c_nvidia(self, c: LayoutTensor, out res: Self.c_reg_tile_type):
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
        """
        Store matrix D to destination memory.

        Stores the result matrix D from tensor core computation to the destination memory.

        Args:
            d_dst: The destination tensor to store the result.
            d_src: The source tensor containing the computed result.
        """

        @parameter
        if is_nvidia_gpu():
            self._store_d_nvidia(d_dst, d_src)
        else:
            self._store_d_amd(d_dst, d_src)

    @always_inline
    fn _store_d_amd(self, d_dst: LayoutTensor, d_src: LayoutTensor):
        constrained[
            d_src.shape[0]() == Self.c_reg_tile_type.shape[0]()
            and d_src.shape[1]() == Self.c_reg_tile_type.shape[1](),
            "src tensor must have the same shape as c_reg_tile_type",
        ]()
        alias mma_m = shape[0]
        alias mma_n = shape[1]
        alias reg_per_thread = num_matrix_reg[mma_m, mma_n]()
        alias warp_layout = Layout.row_major(mma_m // reg_per_thread, mma_n)

        @parameter
        if out_type is DType.float32:
            constrained[
                reg_per_thread in (4, 16),
                "No valid shape to store to LayoutTensor d",
            ]()

            var dst = d_dst.vectorize[4, 1]().distribute[warp_layout](lane_id())
            dst.copy_from(d_src.vectorize[1, 4]())
        else:
            constrained[False, "No valid type to store to LayoutTensor d"]()

    @always_inline
    fn _store_d_nvidia(self, d_dst: LayoutTensor, d_src: LayoutTensor):
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
        out res: Self.c_reg_tile_type,
    ):
        """
        Perform matrix multiply-accumulate operation (MMA).

        Executes `D = A * B + C` using tensor cores.

        Args:
            a: The A matrix input.
            b: The B matrix input.
            c: The C matrix input for accumulation.

        Returns:
            `Self.c_reg_tile_type`: The result of the MMA operation.
        """
        var a_reg = load_to_simd(a)
        var b_reg = load_to_simd(b)
        var c_reg = load_to_simd(c)
        var d_reg = c_reg
        mma(d_reg, a_reg, b_reg, d_reg)
        var d = __type_of(res).stack_allocation()
        d.vectorize[1, Self.c_reg_type.size]()[0, 0] = rebind[
            __type_of(d.vectorize[1, Self.c_reg_type.size]()[0, 0])
        ](d_reg)
        return d

    @always_inline
    fn load_a[
        swizzle: OptionalReg[Swizzle] = None,
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k coordinate of mma tile
    ):
        """
        Load A matrix fragments from shared memory.

        Optimized version for loading A matrix fragments from shared memory.

        Parameters:
            swizzle: Optional memory access pattern for to optimize memory bandwidth.

        Args:
            warp_tile: The source data in shared memory.
            fragments: The destination tensor for fragments.
            mma_tile_coord_k: The K coordinate of the MMA tile. Defaults to 0.
        """
        constrained[
            warp_tile.address_space == AddressSpace.SHARED,
            "warp_tile must be in shared memory",
        ]()

        @parameter
        if is_nvidia_gpu():
            self._load_a_nvidia[swizzle](warp_tile, fragments, mma_tile_coord_k)
        else:
            self._load_a_amd[swizzle](warp_tile, fragments, mma_tile_coord_k)

    @always_inline
    fn _load_a_amd[
        swizzle: OptionalReg[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k corrdinate of mma tile
    ):
        constrained[self.supported_fp32 or self.supported_half]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[warp_tile.dtype]()
        alias num_frags = fragments.shape[0]()
        alias M = shape[0]
        alias K = shape[2]
        alias k_group_size = fragments.element_layout.size() // num_matrix_reg[
            M, K
        ]()

        @parameter
        for i in range(num_frags):
            var mma_tile = warp_tile.tile[M, K * k_group_size](
                i, mma_tile_coord_k
            )
            var a = load_to_simd(self.load_a[swizzle](mma_tile))
            fragments[i, 0] = rebind[frag_type](a)

    @always_inline
    fn _load_a_nvidia[
        swizzle: OptionalReg[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k corrdinate of mma tile
    ):
        constrained[
            self.supported_fp32 or self.supported_half or self.supported_fp8
        ]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[warp_tile.dtype]()
        alias num_frags = fragments.shape[0]()

        var swizzle_offset = mma_tile_coord_k * shape[2] // simd_size

        @parameter
        for i in range(num_frags):
            var mma_tile = warp_tile.tile[shape[0], warp_tile.shape[1]()](i, 0)
            fragments[i, 0] = rebind[frag_type](
                _load_matrix_frag[swizzle](mma_tile, swizzle_offset)
            )

    @always_inline
    fn load_b[
        swizzle: OptionalReg[Swizzle] = None,
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k coordinate of mma tile
        warp_tile_coord_n: UInt = 0,  # n coordinate of warp tile
    ):
        """Load B matrix fragments from shared memory into registers for tensor core operations.

        This function loads matrix B fragments from a warp tile in shared memory into register fragments
        for use in tensor core matrix multiply operations. It handles hardware-specific optimizations
        for both NVIDIA and AMD GPUs.

        Parameters:
            swizzle: Optional memory access pattern for AMD GPUs to optimize memory bandwidth.
                     Must be None when running on NVIDIA GPUs. For NVIDIA GPUs, swizzle is always on.

        Args:
            warp_tile: Source `LayoutTensor` in shared memory containing the B matrix data.
            fragments: Destination `LayoutTensor` to store the loaded matrix fragments.
            mma_tile_coord_k: K-dimension coordinate within the warp tile. Defaults to 0.
            warp_tile_coord_n: N-dimension coordinate within the warp tile. Defaults to 0.

        Note:
            The `warp_tile` must be in shared memory. For NVIDIA GPUs, `swizzle` must be `None`.
            For AMD GPUs, providing an appropriate `swizzle` pattern can improve performance.
        """
        constrained[
            warp_tile.address_space == AddressSpace.SHARED,
            "warp_tile must be in shared memory",
        ]()

        @parameter
        if is_nvidia_gpu():
            constrained[
                swizzle is None, "Swizzle is not supported on NVIDIA for load_b"
            ]()
            self._load_b_nvidia(
                warp_tile, fragments, mma_tile_coord_k, warp_tile_coord_n
            )
        else:
            self._load_b_amd[swizzle](
                warp_tile, fragments, mma_tile_coord_k, warp_tile_coord_n
            )

    @always_inline
    fn _load_b_amd[
        swizzle: OptionalReg[Swizzle],
        *,
    ](
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k corrdinate of mma tile
        warp_tile_coord_n: UInt = 0,  # n coordiante of warp tile
    ):
        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[in_type]()
        alias num_frags = fragments.shape[0]()
        alias N = shape[1]
        alias K = shape[2]
        alias k_group_size = fragments.element_layout.size() // num_matrix_reg[
            N, K
        ]()

        @parameter
        if transpose_b:

            @parameter
            for i in range(num_frags):
                var mma_tile = warp_tile.tile[N, K * k_group_size](
                    i, mma_tile_coord_k
                )
                var frag = load_to_simd(self.load_b[swizzle](mma_tile))
                fragments[i, 0] = rebind[frag_type](frag)
        else:

            @parameter
            for i in range(num_frags):
                var mma_tile = warp_tile.tile[K * k_group_size, N](
                    mma_tile_coord_k, i
                )
                var frag = load_to_simd(self.load_b[swizzle](mma_tile))
                fragments[i, 0] = rebind[frag_type](frag)

    @always_inline
    fn _load_b_nvidia(
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k corrdinate of mma tile
        warp_tile_coord_n: UInt = 0,  # n coordiante of warp tile
    ):
        constrained[
            self.supported_fp32 or self.supported_half or self.supported_fp8
        ]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[in_type]()
        alias num_frags = fragments.shape[0]()
        alias WN = warp_tile.shape[1]()
        alias swizzle = make_ldmatrix_swizzle[
            warp_tile.dtype, warp_tile.stride[0]()
        ]()

        @parameter
        if transpose_b:

            @parameter
            if in_type is DType.float32:
                var swizzle_offset = mma_tile_coord_k * shape[2] // simd_size

                @parameter
                for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag[swizzle=swizzle](
                        mma_tile, swizzle_offset
                    )
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](vec[0], vec[2])
                    )
                    fragments[i + 1, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](vec[1], vec[3])
                    )
            else:
                constrained[
                    self.supported_half or self.supported_fp8,
                    (
                        "Transposed matrix B is only supported for half and fp8"
                        " data types."
                    ),
                ]()

                var swizzle_offset = mma_tile_coord_k * shape[2] // simd_size

                @parameter
                for i in range(0, num_frags, 2):
                    var mma_tile = warp_tile.tile[
                        2 * shape[1], warp_tile.shape[1]()
                    ](i // 2, 0)
                    var vec = _load_matrix_frag[
                        swizzle=swizzle, x4_row_major=True
                    ](mma_tile, swizzle_offset)
                    var high_low = vec.split()
                    fragments[i, 0] = rebind[frag_type](high_low[0])
                    fragments[i + 1, 0] = rebind[frag_type](high_low[1])

        else:

            @parameter
            if in_type is DType.float32:

                @parameter
                for i in range(num_frags):
                    var mma_tile = warp_tile.tile[shape[2], shape[1]](
                        mma_tile_coord_k, i
                    )
                    var frag = mma_tile.distribute[Layout.col_major(4, 8)](
                        lane_id()
                    )
                    fragments[i, 0] = rebind[frag_type](
                        SIMD[warp_tile.dtype, 2](
                            rebind[Scalar[warp_tile.dtype]](frag[0]),
                            rebind[Scalar[warp_tile.dtype]](frag[1]),
                        )
                    )
            elif in_type.is_float8():

                @parameter
                for i in range(num_frags):
                    var mma_tile = warp_tile.tile[shape[2], shape[1]](
                        mma_tile_coord_k, i
                    )
                    var frags = mma_tile.vectorize[4, 1]().distribute[
                        Layout.col_major(4, 8)
                    ](lane_id())
                    fragments[i, 0] = rebind[frag_type](
                        rebind[SIMD[warp_tile.dtype, 4]](frags[0]).join(
                            rebind[SIMD[warp_tile.dtype, 4]](frags[1])
                        ),
                    )

            else:
                constrained[self.supported_half]()

                var mma_tile = warp_tile.tile[shape[2], warp_tile.shape[1]()](
                    mma_tile_coord_k, 0
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
                    var mma_tile_shifted = __type_of(mma_tile)(
                        mma_tile.ptr - warp_tile_coord_n * WN
                    )

                    @parameter
                    for i in range(0, num_frags, 2):
                        var swizzle_offset = (
                            i + warp_tile_coord_n * WN // simd_size
                        )
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True
                        ](mma_tile_shifted, swizzle_offset)
                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])
                else:
                    alias num_frags_round_even = align_down(num_frags, 2)

                    @parameter
                    for i in range(0, num_frags_round_even, 2):
                        # load using x4 layout
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True
                        ](mma_tile, i)

                        var high_low = vec.split()
                        fragments[i, 0] = rebind[frag_type](high_low[0])
                        fragments[i + 1, 0] = rebind[frag_type](high_low[1])

                    @parameter
                    if num_frags % 2:
                        # load using x2 for the last fragment if necessary
                        var vec = _load_matrix_frag[
                            swizzle=swizzle, transposed=True, num_matrices=2
                        ](mma_tile, num_frags_round_even)
                        fragments[num_frags_round_even, 0] = rebind[frag_type](
                            vec
                        )

    @always_inline
    fn load_b(
        self,
        warp_tile: LayoutTensor,
        fragments: LayoutTensor,
        scales: LayoutTensor,
        mma_tile_coord_k: UInt = 0,  # the k coordinate of mma tile
    ):
        """Load quantized B matrix fragments from shared memory with dequantization.

        This function loads int4 quantized matrix B fragments from shared memory, dequantizes them
        using the provided scales, and stores the result in register fragments for tensor core operations.

        Args:
            warp_tile: Source `LayoutTensor` in shared memory containing the quantized B matrix data.
            fragments: Destination `LayoutTensor` to store the dequantized matrix fragments.
            scales: `LayoutTensor` containing the scaling factors for dequantization.
            mma_tile_coord_k: K-dimension coordinate within the warp tile. Defaults to 0.

        Notes:

            - The `warp_tile` must be in shared memory.
            - The `fragments` and `scales` must be in local memory.
            - This function only supports half-precision data types (bfloat16, float16).
            - The quantized data is stored as int4 values packed into int32 elements.
            - Each thread processes multiple fragments by unpacking and dequantizing the int4 values.
        """
        constrained[
            warp_tile.address_space == AddressSpace.SHARED,
            "warp_tile must be in shared memory",
        ]()
        constrained[
            fragments.address_space == AddressSpace.LOCAL,
            "fragments must be in local memory",
        ]()
        constrained[
            scales.address_space == AddressSpace.LOCAL,
            "scales must be in local memory",
        ]()
        constrained[self.supported_half]()

        alias frag_type = fragments.element_type
        alias simd_size = simdwidthof[in_type]()
        alias num_frags = fragments.shape[0]()
        alias pack_factor = 8
        alias repack_tile = Index(64, 16)

        @always_inline
        fn int4tobf16(
            i4: Int32, scale: SIMD[DType.bfloat16, 1]
        ) -> SIMD[DType.bfloat16, 2]:
            alias MASK: Int32 = 0x000F000F
            alias I4s_TO_BF16s_MAGIC_NUM: Int32 = 0x43004300

            alias lut: Int32 = (0xF0 & 0xCC) | 0xAA
            var BF16_BIAS = SIMD[DType.bfloat16, 2](-136, -136)
            var BF16_SCALE = SIMD[DType.bfloat16, 2](scale, scale)
            var BF16_ZERO = SIMD[DType.bfloat16, 2](0, 0)
            var BF16_ONE = SIMD[DType.bfloat16, 2](1, 1)

            var t = lop[lut](i4, MASK, I4s_TO_BF16s_MAGIC_NUM)

            var v = (
                bitcast[DType.bfloat16, 2](t)
                .fma(BF16_ONE, BF16_BIAS)
                .fma(BF16_SCALE, BF16_ZERO)
            )
            return v

        # The wrap_tile is of shape [WK // 64, 128 * n_mma]
        # Every contiguous 128 ints stores a 64x16 repacked tile
        var mma_tile = warp_tile.tile[
            1, (repack_tile[0] * repack_tile[1]) // pack_factor
        ](0, mma_tile_coord_k)

        var vec = bitcast[DType.int32, 4](
            mma_tile.vectorize[1, 4]()[thread_idx.x % WARP_SIZE]
        )

        @parameter
        for i in range(0, num_frags, 2):
            var q_int = vec[i // 2]
            var v1 = int4tobf16(q_int, bitcast[DType.bfloat16, 1](scales[i]))
            q_int >>= 4
            var v2 = int4tobf16(q_int, bitcast[DType.bfloat16, 1](scales[i]))
            fragments[i, 0] = rebind[frag_type](v1.join(v2))
            q_int >>= 4
            v1 = int4tobf16(q_int, bitcast[DType.bfloat16, 1](scales[i + 1]))
            q_int >>= 4
            v2 = int4tobf16(q_int, bitcast[DType.bfloat16, 1](scales[i + 1]))
            fragments[i + 1, 0] = rebind[frag_type](v1.join(v2))

    @always_inline
    fn mma(
        self, a_frag: LayoutTensor, b_frag: LayoutTensor, c_frag: LayoutTensor
    ):
        """Perform matrix multiply-accumulate operation using tensor cores.

        Executes C = A * B + C using tensor cores, where A, B, and C are matrix fragments
        stored in register memory. This function handles the mapping of fragments to
        hardware tensor core operations.

        Args:
            a_frag: Matrix A fragments as a `LayoutTensor`.
            b_frag: Matrix B fragments as a `LayoutTensor`.
            c_frag: Matrix C fragments as a `LayoutTensor` for both input and output.

        Notes:

            - All fragments must be properly loaded using the corresponding load functions.
            - The function assumes fragments are vectorized layout tensors with dimensions num_vectors x 1.
            - The c_frag shape[0] must equal num_m_mmas * num_n_mmas.
            - The result is accumulated in-place in c_frag.
        """
        # TODO: Assume that fragments are all vectorized layout tensor with
        # dims num_vectors x 1. Consider using TensorCore to allocate fragments
        # so the caller don't explicitly maintain the shape.
        alias num_m_mmas = a_frag.shape[0]()
        alias num_n_mmas = b_frag.shape[0]()

        constrained[
            c_frag.shape[0]() == num_m_mmas * num_n_mmas,
            "Fragments size mismatch. Expected c_frag shape[0] to be num_m_mmas"
            " * num_n_mmas = "
            + String(num_m_mmas * num_n_mmas)
            + ", got "
            + String(c_frag.shape[0]()),
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
    swizzle: OptionalReg[Swizzle] = None,
    transposed: Bool = False,
    x4_row_major: Bool = False,
    num_matrices: Int = 4,
    *,
    # Nvidia GPU register is 4B.
    __register_width: Int = 4,
](
    mma_tile: LayoutTensor,
    offset: Int,
    out res: SIMD[
        mma_tile.dtype,
        num_matrices * __register_width // sizeof[mma_tile.dtype](),
    ],
):
    constrained[
        mma_tile.address_space == AddressSpace.SHARED,
        "mma_tile must be shared memory",
    ]()
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
        [8, 2, 2], [num_mat_per_row, 1, 8 * num_mat_per_row]
    ) if x4_row_major else Layout([16, 2], [num_mat_per_row, 1])

    alias ldmatrix_layout = ComposedLayout(
        x4_layout,
        swizzle.value() if swizzle else Swizzle(0, 0, 1),
    )

    var lane_offset = (
        eval_composed[ldmatrix_layout](Int(lane), offset) * simd_size
    )

    return ld_matrix[res.size, transpose=transposed](
        mma_tile.ptr.offset(lane_offset)
    )


@always_inline
fn get_mma_shape[
    input_type: DType, accum_type: DType, shape_id: Int = 0
]() -> IndexList[3]:
    """Returns the appropriate matrix multiply-accumulate (MMA) shape for tensor core operations.

    Selects the optimal MMA shape based on the GPU architecture, input data type,
    accumulation data type, and optional shape identifier. This function handles
    different configurations for both NVIDIA and AMD GPUs.

    Parameters:
        input_type: The data type of the input matrices (A and B).
        accum_type: The data type used for accumulation (C and D).
        shape_id: Optional identifier to select between multiple valid shapes (default: 0).

    Returns:
        An `IndexList[3]` containing the MMA dimensions in the format `[M, N, K]`,
        where `M×N` is the output matrix size and `K` is the reduction dimension.
    """

    @parameter
    if has_nvidia_gpu_accelerator():

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
        elif accum_type is DType.float32 and input_type in (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        ):
            return shape_16x8x32
        else:
            constrained[False, "Unsupported mma shape."]()
            return shape_null
    else:

        @parameter
        if accum_type is DType.float32 and input_type is DType.float32:
            return shape_16x16x4
        elif accum_type is DType.float32 and input_type is DType.bfloat16:
            return shape_16x16x16
        elif accum_type is DType.float32 and input_type is DType.float16:
            return shape_16x16x16
        else:
            constrained[False, "Unsupported mma shape."]()
            return shape_null


@always_inline
fn get_fragment_size[mma_shape: IndexList[3]]() -> IndexList[3]:
    """Calculates the fragment size per thread for a given MMA shape.

    For tensor core operations, each thread in a warp handles a portion of the
    computation. This function determines how many elements each thread needs to
    process for the A, B, and C/D matrices based on the MMA shape.

    Parameters:
        mma_shape: An `IndexList[3]` containing the MMA dimensions [M, N, K].

    Returns:
        An `IndexList[3]` containing the fragment sizes per thread for matrices
        A, B, and C/D respectively, calculated as:
        `[M*K/WARP_SIZE, N*K/WARP_SIZE, M*N/WARP_SIZE]`.
    """
    return IndexList[3](
        mma_shape[0] * mma_shape[2] // WARP_SIZE,
        mma_shape[1] * mma_shape[2] // WARP_SIZE,
        mma_shape[0] * mma_shape[1] // WARP_SIZE,
    )
