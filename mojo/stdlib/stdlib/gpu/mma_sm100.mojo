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
"""This module includes utilities for working with the SM100 MMA instructions."""

from os import abort
from sys import sizeof
from sys._assembly import inlined_assembly
from sys.info import _has_blackwell_tcgen05

from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200, DEFAULT_GPU_ARCH, Info
from gpu.memory import AddressSpace
from memory import UnsafePointer, bitcast

from utils.index import IndexList

# ===----------------------------------------------------------------------=== #
# MMA Instruction Descriptor
# ===----------------------------------------------------------------------=== #


@fieldwise_init("implicit")
@register_passable("trivial")
struct UMMAKind(Stringable, Writable):
    """Struct for UMMA instruction types.

    This struct defines the different types of UMMA instructions that is supported by BlackWell.
    """

    var _value: Int32

    alias KIND_TF32 = Self(0)
    """tf32 type"""

    alias KIND_F16 = Self(2)
    """f16 type"""

    alias KIND_F8F6F4 = Self(3)
    """f8f6f4 type"""

    alias KIND_I8 = Self(4)
    """i8 type"""

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Convert UMMA kind to an integer value.

        Returns:
            The integer value representing the UMMA instruction type.
        """
        return Int(self._value)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Check if two UMMA kinds are equal.

        Args:
            other: The other UMMA kind to compare with.

        Returns:
            True if the UMMA kinds are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Check if two UMMA kinds are not equal.

        Args:
            other: The other UMMA kind to compare with.

        Returns:
            True if the UMMA kinds are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    fn __str__(self) -> String:
        """Convert UMMA kind to a string, this can be used as the instruction qualifier.

        Returns:
            The PTX qualifier representation of the UMMA kind.
        """
        return String.write(self)

    @always_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write the UMMA kind to a writer.

        Parameters:
            W: The writer type that will receive the formatted output.

        Args:
            writer: The writer to write the UMMA kind to.
        """
        if self == Self.KIND_TF32:
            writer.write("kind::tf32")
        elif self == Self.KIND_F16:
            writer.write("kind::f16")
        elif self == Self.KIND_F8F6F4:
            writer.write("kind::f8f6f4")
        elif self == Self.KIND_I8:
            writer.write("kind::i8")
        else:
            writer.write("kind::unknown")


@always_inline
fn _constrained_mma_m[
    mma_m: Int,
    mma_m_valid: Tuple[Int, Int],
    mma_kind: UMMAKind,
    /,
    *,
    use_cta_pair: Bool = False,
]():
    """Constrain the MMA M value based on the MMA valid range and use peer cta flag.

    This function constrains the MMA M value to be within the valid range and returns the constrained value.
    """

    alias using_pair_string = (
        " when using pair cta." if use_cta_pair else " when not using pair cta."
    )

    constrained[
        mma_m in mma_m_valid,
        String(
            "Invalid MMA M: ",
            mma_m,
            " ,MMA M has to be ",
            mma_m_valid[0],
            " or ",
            mma_m_valid[1],
            " for ",
            mma_kind,
            using_pair_string,
        ),
    ]()


@always_inline
fn _constrained_mma_n[
    mma_n: Int,
    mma_n_range: Tuple[Int, Int],
    multiple_of: Int,
    mma_kind: UMMAKind,
    /,
    *,
    use_cta_pair: Bool = False,
]():
    """Constrain the MMA N value based on the MMA valid range and use peer cta flag.

    This function constrains the MMA N value to be within the valid range and returns the constrained value.
    """

    alias using_pair_string = (
        " when using pair cta." if use_cta_pair else " when not using pair cta."
    )

    alias lower_bound = mma_n_range[0]
    alias upper_bound = mma_n_range[1]

    constrained[
        mma_n >= lower_bound
        and mma_n <= upper_bound
        and mma_n % multiple_of == 0,
        String(
            "Invalid MMA N: ",
            mma_n,
            " ,MMA N has to be between ",
            lower_bound,
            " and ",
            upper_bound,
            " and a multiple of ",
            lower_bound % 8,
            " for ",
            mma_kind,
            using_pair_string,
        ),
    ]()


@always_inline
fn _get_f16_mma_shape[
    output_shape: IndexList[2, element_type = DType.uint32],
    /,
    *,
    use_cta_pair: Bool = False,
]() -> IndexList[3, element_type = DType.uint32]:
    """Get the shape of the MMA instruction for F16 MMA kind.

    This function returns the shape of the MMA instruction for F16 MMA kind.
    """
    alias mma_m = output_shape[0]
    alias mma_n = output_shape[1]

    @parameter
    if not use_cta_pair:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_F16,
            use_cta_pair=use_cta_pair,
        ]()

        @parameter
        if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (16, 512),
                16,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()
            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 16)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 32)
        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return abort[IndexList[3, element_type = DType.uint32]](
                "MMA shape not supported."
            )

    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_F16,
            use_cta_pair=use_cta_pair,
        ]()

        @parameter
        if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 16)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 32)
        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return abort[IndexList[3, element_type = DType.uint32]](
                "MMA shape not supported."
            )


@always_inline
fn _get_tf32_mma_shape[
    output_shape: IndexList[2, element_type = DType.uint32],
    /,
    *,
    use_pair_cta: Bool = False,
]() -> IndexList[3, element_type = DType.uint32]:
    """Get the shape of the MMA instruction for TF32 MMA kind.

    This function returns the shape of the MMA instruction for TF32 MMA kind.
    """

    alias mma_m = output_shape[0]
    alias mma_n = output_shape[1]

    @parameter
    if not use_pair_cta:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_TF32,
            use_cta_pair=use_pair_cta,
        ]()

        @parameter
        if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 8)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 512),
                16,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 16)
        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return abort[IndexList[3, element_type = DType.uint32]](
                "MMA shape not supported."
            )
    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_TF32,
            use_cta_pair=use_pair_cta,
        ]()

        @parameter
        if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 512),
                16,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 8)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 16)
        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return abort[IndexList[3, element_type = DType.uint32]](
                "MMA shape not supported."
            )


@always_inline
fn _get_f8f6f4_mma_shape[
    output_shape: IndexList[2, element_type = DType.uint32],
    /,
    *,
    use_pair_cta: Bool = False,
]() -> IndexList[3, element_type = DType.uint32]:
    """Get the shape of the MMA instruction for F8F6F4 MMA kind.

    This function returns the shape of the MMA instruction for F8F6F4 MMA kind.
    """

    alias mma_m = output_shape[0]
    alias mma_n = output_shape[1]

    @parameter
    if not use_pair_cta:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_F8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        @parameter
        if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 32)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 512),
                16,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 64)

        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return abort[IndexList[3, element_type = DType.uint32]](
                "MMA shape not supported."
            )

    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_F8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        @parameter
        if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 32)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 512),
                32,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type = DType.uint32](mma_m, mma_n, 64)
        else:
            constrained[False, String("Invalid MMA shape: ", mma_m, mma_n)]()

            return IndexList[3, element_type = DType.uint32](0, 0, 0)


@register_passable("trivial")
struct UMMAInsDescriptor[
    mma_kind: UMMAKind,
]:
    """Descriptor for UMMA instructions.

    This struct represents a descriptor that encodes information about UMMA instructions.
    The descriptor contains the following bit fields:

    - Sparsity (2 bits): The sparsity of the input matrices. Currently defaults to dense matrices.
    - Saturate for integer types (1 bits): Whether to saturate the result for integer types. Currently not supported.
    - Matrix D type (2 bits): Data type of matrix D.
    - Matrix A type (3 bits): Data type of matrix A.
    - Matrix B type (3 bits): Data type of matrix B.
    - Negate A matrix (1 bit): Whether to negate matrix A. Currently defaults to False.
    - Negate B matrix (1 bit): Whether to negate matrix B. Currently defaults to False.
    - Transpose A (1 bit): Whether to transpose matrix A.
    - Transpose B (1 bit): Whether to transpose matrix B.
    - N, Dimension of Matrix B (6 bits): Number of columns in matrix B. 3 LSBs are unused.
    - M, Dimension of Matrix A (6 bits): Number of rows in matrix A. 3 LSBs are unused.

    Parameters:
        mma_kind: The kind of UMMA instruction.

    See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520mma#tcgen05-instuction-desc-kind-tf32-f16-f8f6f4
    """

    var desc: UInt32
    """The 32-bit descriptor value that encodes UMMA instruction information.

    This field stores the complete descriptor with all bit fields packed into a single 32-bit integer:
    - Bits 0-1: Sparsity selector(2 bits)
    - Bits 2: Sparsity enable(1 bit)
    - Bits 3: Saturate for integer types (1 bit)
    - Bits 4-5: Matrix D type (2 bits)
    - Bits 6: Reserved (1 bit)
    - Bits 7-9: Matrix A type (3 bits)
    - Bits 10-12: Matrix B type (3 bits)
    - Bits 13: Negate A matrix (1 bit)
    - Bits 14: Negate B matrix (1 bit)
    - Bits 15: Transpose A (1 bit)
    - Bits 16: Transpose B (1 bit)
    - Bits 17-22: N, Dimension of Matrix B (6 bits)
    - Bits 23: Reserved (1 bit)
    - Bits 24-28: M, Dimension of Matrix A (5 bits)
    - Bits 29: Reserved (1 bit)
    - Bits 30-31: Maximum shift while attempting B matrix (2 bits)
    """

    @implicit
    fn __init__(out self, value: UInt32):
        """Initialize descriptor with raw 32-bit value.

        This constructor allows creating a descriptor directly from a 32-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        Args:
            value: A 32-bit integer containing the complete descriptor bit layout.
        """
        self.desc = value

    @staticmethod
    fn _insert_bit[start_bit: Int](desc: UInt32, val: UInt32) -> UInt32:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            desc: Descriptor value.
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """

        return desc | (val << start_bit)

    @staticmethod
    fn _create_tf32_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for TF32 UMMA instructions.

        This function creates a descriptor for TF32 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        constrained[
            d_type == DType.float32
            and a_type == DType.float32
            and b_type == DType.float32,
            String(
                "Invalid operand data type for UMMA instruction: ",
                d_type,
                " and ",
                a_type,
                " and ",
                b_type,
            ),
        ]()

        alias d_type_bit = Self._insert_bit[4](0x0, 0x1)
        alias a_type_bit = Self._insert_bit[7](d_type_bit, 0x2)
        alias desc = Self._insert_bit[10](a_type_bit, 0x2)

        return desc

    @staticmethod
    fn _create_f16_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for F16 UMMA instructions.

        This function creates a descriptor for F16 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        alias available_d_types = (DType.float32, DType.float16)
        alias available_operand_types = (DType.bfloat16, DType.float16)

        constrained[
            d_type in available_d_types,
            String("Invalid d data type for UMMA instruction: ", d_type),
        ]()

        constrained[
            a_type in available_operand_types
            and b_type in available_operand_types,
            String(
                "Invalid operand data type for UMMA instruction: ",
                a_type,
                " and ",
                b_type,
            ),
        ]()

        alias d_type_bit = Self._insert_bit[4](
            0x0, 1 if d_type == DType.float32 else 0
        )
        alias a_type_bit = Self._insert_bit[7](
            d_type_bit, 1 if a_type == DType.bfloat16 else 0
        )
        alias desc = Self._insert_bit[10](
            a_type_bit, 1 if b_type == DType.bfloat16 else 0
        )

        return desc

    @staticmethod
    fn _create_f8f6f4_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for F8F6F4 UMMA instructions.

        This function creates a descriptor for F8F6F4 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        alias available_d_types = (DType.float16, DType.float32)
        alias available_operand_types = (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        )

        constrained[
            d_type in available_d_types,
            String("Invalid d data type for UMMA instruction: ", d_type),
        ]()

        constrained[
            a_type in available_operand_types
            and b_type in available_operand_types,
            String(
                "Currently only support E4M3 and E5M2 for UMMA kind: ",
                mma_kind,
            ),
        ]()

        alias d_type_bit = Self._insert_bit[4](
            0x0, 1 if d_type == DType.float32 else 0
        )

        alias a_type_bit = Self._insert_bit[7](
            d_type_bit, 1 if a_type == DType.float8_e5m2 else 0
        )
        alias desc = Self._insert_bit[10](
            a_type_bit, 1 if b_type == DType.float8_e5m2 else 0
        )

        return desc

    @staticmethod
    fn create[
        d_type: DType,
        a_type: DType,
        b_type: DType,
        output_shape: IndexList[2, element_type = DType.uint32],
        /,
        *,
        transpose_a: Bool = False,
        transpose_b: Bool = True,
    ]() -> Self:
        """Create a descriptor for UMMA instructions.

        This function creates a descriptor for UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.
            output_shape: The shape of the output matrix.
            transpose_a: Whether to transpose matrix A.
            transpose_b: Whether to transpose matrix B.

        Returns:
            A 32-bit integer containing the complete descriptor bit layout.
        """

        alias M_bit = Self._insert_bit[17](0x0, output_shape[1] >> 3)
        alias desc = Self._insert_bit[24](M_bit, output_shape[0] >> 4)

        alias transpose_a_bit = Self._insert_bit[15](
            0x0, 1 if transpose_a else 0
        )
        alias transpose_bit = Self._insert_bit[16](
            transpose_a_bit, 0 if transpose_b else 1
        )

        @parameter
        if mma_kind == UMMAKind.KIND_TF32:
            return (
                desc
                | Self._create_tf32_desc[d_type, a_type, b_type]()
                | transpose_bit
            )

        @parameter
        if mma_kind == UMMAKind.KIND_F16:
            return (
                desc
                | Self._create_f16_desc[d_type, a_type, b_type]()
                | transpose_bit
            )

        @parameter
        if mma_kind == UMMAKind.KIND_F8F6F4:
            return (
                desc
                | Self._create_f8f6f4_desc[d_type, a_type, b_type]()
                | transpose_bit
            )

        constrained[False, String("Unsupported UMMA kind: ", mma_kind)]()

        return Self(0x0)


# ===----------------------------------------------------------------------=== #
# MMA Shared Memory Descriptor
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct MMASmemDescriptor:
    """Descriptor for shared memory operands tcgen05 mma instructions.

    This struct represents a descriptor that encodes information about shared memory layout
    and access patterns for warp group matrix multiply operations. The descriptor contains
    the following bit fields:

    bits layout:

    Bit-field | size | Description
       0-13   |  14  | Base address in shared memory
      14-15   |   2  | Unused, 0
      16-29   |  14  | LBO: leading dim byte offset
      30-31   |   2  | Unused, 0
      32-45   |  14  | SBO: stride dim byte offset
      46-48   |   3  | Unused, 0
      49-51   |   3  | Matrix Base offset, 0 for canonical layouts
      52      |   1  | LBO mode, only matters for 48B K tile
      53-60   |   8  | fixed, 0
      61-63   |   3  | Swizzle mode

    - Start address, LBO, SBO ignores 4 LSBs.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-desc-layout

    """

    var desc: UInt64
    """The 64-bit descriptor encodes shared memory operand information."""

    @implicit
    @always_inline
    fn __init__(out self, val: UInt64):
        """Initialize descriptor with raw 64-bit value.

        This constructor allows creating a descriptor directly from a 64-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        The implicit attribute enables automatic conversion from `UInt64` to `MMASmemDescriptor`.

        Args:
            val: A 64-bit integer containing the complete descriptor bit layout.
        """
        self.desc = val

    @always_inline
    fn _insert_bit[start_bit: Int](self, val: UInt64) -> UInt64:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """
        return self.desc | (val << start_bit)

    @staticmethod
    @always_inline
    fn create[
        stride_byte_offset: Int,
        leading_byte_offset: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](
        smem_ptr: UnsafePointer[
            _, address_space = AddressSpace.SHARED, *_, **_
        ],
    ) -> Self:
        """Create a descriptor for shared memory operand.

        Parameters:
            stride_byte_offset: Stride dimension offset in bytes.
            leading_byte_offset: Leading dimension stride in bytes.
            swizzle_mode: Memory access pattern mode.

        Args:
            smem_ptr: Pointer to shared memory operand.

        Returns:
            Initialized descriptor for the shared memory operand.
        """

        # TMA enumerates no swizzle, 32, 64, 128B as 0, 1, 2, 3.
        # WGMMA enumerates these as 0, 3, 2, 1.
        @parameter
        fn _convert_swizzle_enum[mode: TensorMapSwizzle]() -> Int64:
            @parameter
            if mode == TensorMapSwizzle.SWIZZLE_NONE:
                return 0
            elif mode == TensorMapSwizzle.SWIZZLE_32B:
                return 6
            elif mode == TensorMapSwizzle.SWIZZLE_64B:
                return 4
            elif mode == TensorMapSwizzle.SWIZZLE_128B:
                return 2
            else:
                constrained[False, String("Unsupported swizzle mode: ", mode)]()
                return 0

        alias swizzle = _convert_swizzle_enum[swizzle_mode._value]()

        # Extract 18 bits and ignore 4 LSB.
        var base_ptr = UInt32(Int(smem_ptr))
        var start_address = UInt64((base_ptr & 0x3FFFF) >> 4)

        # Ignore 4 LSB.
        var sbo = UInt64((stride_byte_offset & 0x3FFF) >> 4)
        var lbo = UInt64((leading_byte_offset & 0x3FFF) >> 4)

        # Start from LSB. Mask out higher bits to avoid overwriting.
        var desc = UInt64(0)
        # bits  0-13 address in share memory
        desc = Self._insert_bit[0](desc, start_address)
        # bits 14-16 unused
        # bits 16-29 leading dim byte offset
        desc = Self._insert_bit[16](desc, lbo)
        # bits 30-32 unused
        # bits 32-45 stride dim byte offset
        desc = Self._insert_bit[32](desc, sbo)
        # bits 46-48 001
        desc = Self._insert_bit[46](desc, 1)
        # bits 49-51 matrix base offset, not supported
        # bits 52    LBO mode, only matters for 48B K tile and not supported
        # bits 53-60 fixed, 0
        # bits 61-63 swizzle type
        desc = Self._insert_bit[61](desc, UInt64(swizzle))

        return desc

    @always_inline
    fn __iadd__(mut self, offset: Int):
        """Add offset to descriptor's base address in-place.

        Args:
            offset: Byte offset to add to base address.
        """
        self = self + offset

    @always_inline
    fn __add__(self, offset: Int) -> Self:
        """Add offset to descriptor's base address.

        Args:
            offset: Byte offset to add to base address.

        Returns:
            New descriptor with updated base address.
        """
        return self.desc + ((offset & 0x3FFFF) >> 4)


# ===----------------------------------------------------------------------=== #
# UMMA
# ===----------------------------------------------------------------------=== #


@always_inline
fn mma[
    kind: UMMAKind, //,
    cta_group: Int = 1,
    /,
    *,
    c_scale: UInt32 = 1,
](
    a_desc: MMASmemDescriptor,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.
        c_scale: Scale factor for the C matrix, 0 or 1.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
    """
    constrained[
        _has_blackwell_tcgen05(), "tcgen05.mma not supported on this GPU"
    ]()

    constrained[
        c_scale == 0 or c_scale == 1, String("Invalid c_scale: ", c_scale)
    ]()

    @parameter
    if cta_group == 1:
        var masks = IndexList[4, element_type = DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,n,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type = DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,n,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        constrained[False, String("Unsupported cta group: ", cta_group)]()


@always_inline
fn mma[
    kind: UMMAKind, //,
    cta_group: Int = 1,
    /,
    *,
    c_scale: UInt32 = 1,
](
    a_desc: UInt32,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.
        c_scale: Scale factor for the C matrix, 0 or 1.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
    """
    constrained[
        _has_blackwell_tcgen05(), "tcgen05.mma not supported on this GPU"
    ]()

    constrained[
        c_scale == 0 or c_scale == 1, String("Invalid c_scale: ", c_scale)
    ]()

    @parameter
    if cta_group == 1:
        var masks = IndexList[4, element_type = DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,n,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type = DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,n,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        constrained[False, String("Unsupported cta group: ", cta_group)]()


# ===----------------------------------------------------------------------=== #
# UMMA Sync
# ===----------------------------------------------------------------------=== #


@always_inline
fn mma_arrive[
    cta_group: Int = 1,
](mbar_ptr: UnsafePointer[address_space = AddressSpace.SHARED, *_, **_],):
    """Arrive at the mbar pointer for the MMA instruction.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
    """

    constrained[
        cta_group in (1, 2),
        String("Unsupported cta group: ", cta_group),
    ]()

    alias type = mbar_ptr.type
    constrained[sizeof[type]() == 8, "mbar_ptr must be 8 bytes"]()

    inlined_assembly[
        "tcgen05.commit.cta_group::"
        + String(cta_group)
        + ".mbarrier::arrive::one.shared::cluster.b64 [$0];",
        NoneType,
        constraints="r",
    ](Int32(Int(mbar_ptr)))


@always_inline
fn mma_arrive_multicast[
    cta_group: Int = 1,
](
    mbar_ptr: UnsafePointer[address_space = AddressSpace.SHARED, *_, **_],
    cta_mask: UInt16,
):
    """Arrive at the mbar pointer for the MMA instruction for multiple ctas.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
        cta_mask: Mask of ctas to signal.
    """

    constrained[
        cta_group in (1, 2),
        String("Unsupported cta group: ", cta_group),
    ]()

    alias type = mbar_ptr.type
    constrained[sizeof[type]() == 8, "mbar_ptr must be 8 bytes"]()

    inlined_assembly[
        "tcgen05.commit.cta_group::"
        + String(cta_group)
        + ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [$0], $1;",
        NoneType,
        constraints="r,h",
    ](Int32(Int(mbar_ptr)), cta_mask)
