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


@value
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

    @implicit
    fn __init__(out self, value: Int32):
        """Initialize UMMA kind with an integer value.

        Args:
            value: The integer value representing the UMMA instruction type.
        """
        self._value = value

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
        transpose_b: Bool = False,
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
            transpose_a_bit, 1 if transpose_b else 0
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
