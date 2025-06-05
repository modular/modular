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

from os import abort
from pathlib import Path
from sys.ffi import _find_dylib
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function
from sys.ffi import _Global, _OwnedDLHandle

from gpu.host._nvidia_cuda import _CUstream_st
from memory import UnsafePointer

from utils import StaticTuple

from .cublas import ComputeType
from .dtype import DataType, Property
from .result import Result

alias Context = NoneType

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias CUDA_CUBLASLT_LIBRARY_PATHS = List[Path](
    "libcublasLt.so.12",
    "/usr/local/cuda/lib64/libcublasLt.so.12",
)

alias CUDA_CUBLASLT_LIBRARY = _Global[
    "CUDA_CUBLASLT_LIBRARY", _OwnedDLHandle, _init_dylib
]


fn _init_dylib() -> _OwnedDLHandle:
    return _find_dylib(CUDA_CUBLASLT_LIBRARY_PATHS)


@always_inline
fn _get_dylib_function[
    func_name: StaticString, result_type: AnyTrivialRegType
]() -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLASLT_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


fn cublasLtMatmulAlgoConfigSetAttribute(
    algo: UnsafePointer[MatmulAlgorithm],
    attr: AlgorithmConfig,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) -> Result:
    """Set algo configuration attribute.

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoConfigSetAttribute",
        fn (
            UnsafePointer[MatmulAlgorithm],
            AlgorithmConfig,
            UnsafePointer[NoneType],
            Int,
        ) -> Result,
    ]()(algo, attr, buf, size_in_bytes)


fn cublasLtCreate(
    light_handle: UnsafePointer[UnsafePointer[Context]],
) -> Result:
    return _get_dylib_function[
        "cublasLtCreate",
        fn (UnsafePointer[UnsafePointer[Context]]) -> Result,
    ]()(light_handle)


fn cublasLtMatrixTransformDescCreate(
    transform_desc: UnsafePointer[UnsafePointer[Transform]],
    scale_type: DataType,
) -> Result:
    """Create new matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescCreate",
        fn (
            UnsafePointer[UnsafePointer[Transform]],
            DataType,
        ) -> Result,
    ]()(transform_desc, scale_type)


@fieldwise_init
@register_passable("trivial")
struct Order:
    """Enum for data ordering ."""

    var _value: Int32
    alias COL = Self(0)
    """Column-major

    Leading dimension is the stride (in elements) to the beginning of next column in memory.
    """
    alias ROW = Self(1)
    """Row major

    Leading dimension is the stride (in elements) to the beginning of next row in memory.
    """
    alias COL32 = Self(2)
    """Column-major ordered tiles of 32 columns.

    Leading dimension is the stride (in elements) to the beginning of next group of 32-columns. E.g. if matrix has 33
    columns and 2 rows, ld must be at least (32) * 2 = 64.
    """
    alias COL4_4R2_8C = Self(3)
    """Column-major ordered tiles of composite tiles with total 32 columns and 8 rows, tile composed of interleaved
    inner tiles of 4 columns within 4 even or odd rows in an alternating pattern.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32 * 8) * 1 = 256.
    """
    alias COL32_2R_4R4 = Self(4)
    """Column-major ordered tiles of composite tiles with total 32 columns ands 32 rows.
    Element offset within the tile is calculated as (((row%8)/2*4+row/8)*2+row%2)*32+col.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 32 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32*32)*1 = 1024.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.COL:
            return "COL"
        if self == Self.ROW:
            return "ROW"
        if self == Self.COL32:
            return "COL32"
        if self == Self.COL4_4R2_8C:
            return "COL4_4R2_8C"
        if self == Self.COL32_2R_4R4:
            return "COL32_2R_4R4"
        return abort[String]("invalid Order entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtMatrixLayoutSetAttribute(
    mat_layout: UnsafePointer[MatrixLayout],
    attr: LayoutAttribute,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) -> Result:
    """Set matrix layout descriptor attribute.

    matLayout    The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutSetAttribute",
        fn (
            UnsafePointer[MatrixLayout],
            LayoutAttribute,
            UnsafePointer[NoneType],
            Int,
        ) -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes)


@fieldwise_init
@register_passable("trivial")
struct ClusterShape:
    """Thread Block Cluster size.

    Typically dimensioned similar to Tile, with the third coordinate unused at this time.
    ."""

    var _value: Int32
    alias SHAPE_AUTO = Self(0)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x1x1 = Self(1)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x1x1 = Self(2)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x1x1 = Self(3)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x2x1 = Self(4)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x2x1 = Self(5)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x2x1 = Self(6)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x4x1 = Self(7)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x4x1 = Self(8)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x4x1 = Self(9)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_8x1x1 = Self(10)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x8x1 = Self(11)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_8x2x1 = Self(12)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x8x1 = Self(13)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_16x1x1 = Self(14)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x16x1 = Self(15)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x1x1 = Self(16)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x1x1 = Self(17)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_6x1x1 = Self(18)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_7x1x1 = Self(19)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_9x1x1 = Self(20)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_10x1x1 = Self(21)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_11x1x1 = Self(22)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_12x1x1 = Self(23)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_13x1x1 = Self(24)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_14x1x1 = Self(25)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_15x1x1 = Self(26)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x2x1 = Self(27)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x2x1 = Self(28)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_6x2x1 = Self(29)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_7x2x1 = Self(30)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x3x1 = Self(31)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x3x1 = Self(32)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x3x1 = Self(33)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x3x1 = Self(34)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x3x1 = Self(35)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x4x1 = Self(36)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x5x1 = Self(37)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x5x1 = Self(38)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x5x1 = Self(39)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x6x1 = Self(40)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x6x1 = Self(41)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x7x1 = Self(42)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x7x1 = Self(43)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x9x1 = Self(44)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x10x1 = Self(45)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x11x1 = Self(46)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x12x1 = Self(47)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x13x1 = Self(48)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x14x1 = Self(49)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x15x1 = Self(50)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_END = Self(51)
    """Let library pick cluster shape automatically.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.SHAPE_AUTO:
            return "SHAPE_AUTO"
        if self == Self.SHAPE_1x1x1:
            return "SHAPE_1x1x1"
        if self == Self.SHAPE_2x1x1:
            return "SHAPE_2x1x1"
        if self == Self.SHAPE_4x1x1:
            return "SHAPE_4x1x1"
        if self == Self.SHAPE_1x2x1:
            return "SHAPE_1x2x1"
        if self == Self.SHAPE_2x2x1:
            return "SHAPE_2x2x1"
        if self == Self.SHAPE_4x2x1:
            return "SHAPE_4x2x1"
        if self == Self.SHAPE_1x4x1:
            return "SHAPE_1x4x1"
        if self == Self.SHAPE_2x4x1:
            return "SHAPE_2x4x1"
        if self == Self.SHAPE_4x4x1:
            return "SHAPE_4x4x1"
        if self == Self.SHAPE_8x1x1:
            return "SHAPE_8x1x1"
        if self == Self.SHAPE_1x8x1:
            return "SHAPE_1x8x1"
        if self == Self.SHAPE_8x2x1:
            return "SHAPE_8x2x1"
        if self == Self.SHAPE_2x8x1:
            return "SHAPE_2x8x1"
        if self == Self.SHAPE_16x1x1:
            return "SHAPE_16x1x1"
        if self == Self.SHAPE_1x16x1:
            return "SHAPE_1x16x1"
        if self == Self.SHAPE_3x1x1:
            return "SHAPE_3x1x1"
        if self == Self.SHAPE_5x1x1:
            return "SHAPE_5x1x1"
        if self == Self.SHAPE_6x1x1:
            return "SHAPE_6x1x1"
        if self == Self.SHAPE_7x1x1:
            return "SHAPE_7x1x1"
        if self == Self.SHAPE_9x1x1:
            return "SHAPE_9x1x1"
        if self == Self.SHAPE_10x1x1:
            return "SHAPE_10x1x1"
        if self == Self.SHAPE_11x1x1:
            return "SHAPE_11x1x1"
        if self == Self.SHAPE_12x1x1:
            return "SHAPE_12x1x1"
        if self == Self.SHAPE_13x1x1:
            return "SHAPE_13x1x1"
        if self == Self.SHAPE_14x1x1:
            return "SHAPE_14x1x1"
        if self == Self.SHAPE_15x1x1:
            return "SHAPE_15x1x1"
        if self == Self.SHAPE_3x2x1:
            return "SHAPE_3x2x1"
        if self == Self.SHAPE_5x2x1:
            return "SHAPE_5x2x1"
        if self == Self.SHAPE_6x2x1:
            return "SHAPE_6x2x1"
        if self == Self.SHAPE_7x2x1:
            return "SHAPE_7x2x1"
        if self == Self.SHAPE_1x3x1:
            return "SHAPE_1x3x1"
        if self == Self.SHAPE_2x3x1:
            return "SHAPE_2x3x1"
        if self == Self.SHAPE_3x3x1:
            return "SHAPE_3x3x1"
        if self == Self.SHAPE_4x3x1:
            return "SHAPE_4x3x1"
        if self == Self.SHAPE_5x3x1:
            return "SHAPE_5x3x1"
        if self == Self.SHAPE_3x4x1:
            return "SHAPE_3x4x1"
        if self == Self.SHAPE_1x5x1:
            return "SHAPE_1x5x1"
        if self == Self.SHAPE_2x5x1:
            return "SHAPE_2x5x1"
        if self == Self.SHAPE_3x5x1:
            return "SHAPE_3x5x1"
        if self == Self.SHAPE_1x6x1:
            return "SHAPE_1x6x1"
        if self == Self.SHAPE_2x6x1:
            return "SHAPE_2x6x1"
        if self == Self.SHAPE_1x7x1:
            return "SHAPE_1x7x1"
        if self == Self.SHAPE_2x7x1:
            return "SHAPE_2x7x1"
        if self == Self.SHAPE_1x9x1:
            return "SHAPE_1x9x1"
        if self == Self.SHAPE_1x10x1:
            return "SHAPE_1x10x1"
        if self == Self.SHAPE_1x11x1:
            return "SHAPE_1x11x1"
        if self == Self.SHAPE_1x12x1:
            return "SHAPE_1x12x1"
        if self == Self.SHAPE_1x13x1:
            return "SHAPE_1x13x1"
        if self == Self.SHAPE_1x14x1:
            return "SHAPE_1x14x1"
        if self == Self.SHAPE_1x15x1:
            return "SHAPE_1x15x1"
        if self == Self.SHAPE_END:
            return "SHAPE_END"
        return abort[String]("invalid ClusterShape entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtHeuristicsCacheSetCapacity(capacity: Int) -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheSetCapacity", fn (Int) -> Result
    ]()(capacity)


@register_passable("trivial")
struct MatmulAlgorithmCapability:
    """Capabilities Attributes that can be retrieved from an initialized Algo structure
    ."""

    var _value: Int32
    alias SPLITK_SUPPORT = Self(0)
    """support for split K, see SPLITK_NUM

    int32_t, 0 means no support, supported otherwise.
    """
    alias REDUCTION_SCHEME_MASK = Self(1)
    """reduction scheme mask, see ReductionScheme; shows supported reduction schemes, if reduction scheme is
    not masked out it is supported.

    e.g. int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & COMPUTE_TYPE) ==
    COMPUTE_TYPE ? 1 : 0;

    uint32_t.
    """
    alias CTA_SWIZZLING_SUPPORT = Self(2)
    """support for cta swizzling, see CTA_SWIZZLING

    uint32_t, 0 means no support, 1 means supported value of 1, other values are reserved.
    """
    alias STRIDED_BATCH_SUPPORT = Self(3)
    """support strided batch

    int32_t, 0 means no support, supported otherwise.
    """
    alias OUT_OF_PLACE_RESULT_SUPPORT = Self(4)
    """support results out of place (D != C in D = alpha.A.B + beta.C)

    int32_t, 0 means no support, supported otherwise.
    """
    alias UPLO_SUPPORT = Self(5)
    """syrk/herk support (on top of regular gemm)

    int32_t, 0 means no support, supported otherwise.
    """
    alias TILE_IDS = Self(6)
    """tile ids possible to use, see Tile; if no tile ids are supported use
    TILE_UNDEFINED

    use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count

    array of uint32_t.
    """
    alias CUSTOM_OPTION_MAX = Self(7)
    """custom option range is from 0 to CUSTOM_OPTION_MAX (inclusive), see
    CUSTOM_OPTION

    int32_t.
    """
    alias CUSTOM_MEMORY_ORDER = Self(8)
    """whether algorithm supports custom (not COL or ROW memory order), see Order

    int32_t 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different
    requirements;.
    """
    alias POINTER_MODE_MASK = Self(9)
    """bitmask enumerating pointer modes algorithm supports

    uint32_t, see PointerModeMask.
    """
    alias EPILOGUE_MASK = Self(10)
    """bitmask enumerating kinds of postprocessing algorithm supports in the epilogue

    uint32_t, see Epilogue.
    """
    alias STAGES_IDS = Self(11)
    """stages ids possible to use, see Stages; if no stages ids are supported use
    STAGES_UNDEFINED

    use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count

    array of uint32_t.
    """
    alias LD_NEGATIVE = Self(12)
    """support for nagative ld for all of the matrices

    int32_t 0 means no support, supported otherwise.
    """
    alias NUMERICAL_IMPL_FLAGS = Self(13)
    """details about algorithm's implementation that affect it's numerical behavior

    uint64_t, see cublasLtNumericalImplFlags_t.
    """
    alias MIN_ALIGNMENT_A_BYTES = Self(14)
    """minimum alignment required for A matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_B_BYTES = Self(15)
    """minimum alignment required for B matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_C_BYTES = Self(16)
    """minimum alignment required for C matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_D_BYTES = Self(17)
    """minimum alignment required for D matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias ATOMIC_SYNC = Self(18)
    """EXPERIMENTAL: support for synchronization via atomic counters

    int32_t.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.SPLITK_SUPPORT:
            return "SPLITK_SUPPORT"
        if self == Self.REDUCTION_SCHEME_MASK:
            return "REDUCTION_SCHEME_MASK"
        if self == Self.CTA_SWIZZLING_SUPPORT:
            return "CTA_SWIZZLING_SUPPORT"
        if self == Self.STRIDED_BATCH_SUPPORT:
            return "STRIDED_BATCH_SUPPORT"
        if self == Self.OUT_OF_PLACE_RESULT_SUPPORT:
            return "OUT_OF_PLACE_RESULT_SUPPORT"
        if self == Self.UPLO_SUPPORT:
            return "UPLO_SUPPORT"
        if self == Self.TILE_IDS:
            return "TILE_IDS"
        if self == Self.CUSTOM_OPTION_MAX:
            return "CUSTOM_OPTION_MAX"
        if self == Self.CUSTOM_MEMORY_ORDER:
            return "CUSTOM_MEMORY_ORDER"
        if self == Self.POINTER_MODE_MASK:
            return "POINTER_MODE_MASK"
        if self == Self.EPILOGUE_MASK:
            return "EPILOGUE_MASK"
        if self == Self.STAGES_IDS:
            return "STAGES_IDS"
        if self == Self.LD_NEGATIVE:
            return "LD_NEGATIVE"
        if self == Self.NUMERICAL_IMPL_FLAGS:
            return "NUMERICAL_IMPL_FLAGS"
        if self == Self.MIN_ALIGNMENT_A_BYTES:
            return "MIN_ALIGNMENT_A_BYTES"
        if self == Self.MIN_ALIGNMENT_B_BYTES:
            return "MIN_ALIGNMENT_B_BYTES"
        if self == Self.MIN_ALIGNMENT_C_BYTES:
            return "MIN_ALIGNMENT_C_BYTES"
        if self == Self.MIN_ALIGNMENT_D_BYTES:
            return "MIN_ALIGNMENT_D_BYTES"
        if self == Self.ATOMIC_SYNC:
            return "ATOMIC_SYNC"
        return abort[String]("invalid MatmulAlgorithmCapability entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtGetStatusString(status: Result) raises -> UnsafePointer[Int8]:
    return _get_dylib_function[
        "cublasLtGetStatusString", fn (Result) raises -> UnsafePointer[Int8]
    ]()(status)


@fieldwise_init
@register_passable("trivial")
struct PointerMode:
    """UnsafePointer mode to use for alpha/beta ."""

    var _value: Int32
    alias HOST = PointerMode(0)
    """matches CUBLAS_POINTER_MODE_HOST, pointer targets a single value host memory.
    """
    alias DEVICE = PointerMode(1)
    """matches CUBLAS_POINTER_MODE_DEVICE, pointer targets a single value device memory.
    """
    alias DEVICE_VECTOR = PointerMode(2)
    """pointer targets an array in device memory.
    """
    alias ALPHA_DEVICE_VECTOR_BETA_ZERO = PointerMode(3)
    """alpha pointer targets an array in device memory, beta is zero. Note:
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE is not supported, must be 0.
    """
    alias ALPHA_DEVICE_VECTOR_BETA_HOST = PointerMode(4)
    """alpha pointer targets an array in device memory, beta is a single value in host memory.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.HOST:
            return "HOST"
        if self == Self.DEVICE:
            return "DEVICE"
        if self == Self.DEVICE_VECTOR:
            return "DEVICE_VECTOR"
        if self == Self.ALPHA_DEVICE_VECTOR_BETA_ZERO:
            return "ALPHA_DEVICE_VECTOR_BETA_ZERO"
        if self == Self.ALPHA_DEVICE_VECTOR_BETA_HOST:
            return "ALPHA_DEVICE_VECTOR_BETA_HOST"
        return abort[String]("invalid PointerMode entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtMatmulDescGetAttribute(
    matmul_desc: UnsafePointer[Descriptor],
    attr: cublasLtMatmulDescAttributes_t,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get matmul operation descriptor attribute.

    matmulDesc   The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescGetAttribute",
        fn (
            UnsafePointer[Descriptor],
            cublasLtMatmulDescAttributes_t,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes, size_written)


# Opaque descriptor for matrix memory layout
# .
alias cublasLtMatrixLayout_t = UnsafePointer[MatrixLayout]

# Opaque descriptor for cublasLtMatrixTransform() operation details
# .
alias cublasLtMatrixTransformDesc_t = UnsafePointer[Transform]


fn cublasLtMatmulAlgoCheck(
    light_handle: UnsafePointer[Context],
    operation_desc: UnsafePointer[Descriptor],
    _adesc: UnsafePointer[MatrixLayout],
    _bdesc: UnsafePointer[MatrixLayout],
    _cdesc: UnsafePointer[MatrixLayout],
    _ddesc: UnsafePointer[MatrixLayout],
    algo: UnsafePointer[MatmulAlgorithm],
    result: UnsafePointer[cublasLtMatmulHeuristicResult_t],
) -> Result:
    """Check configured algo descriptor for correctness and support on current device.

    Result includes required workspace size and calculated wave count.

    CUBLAS_STATUS_SUCCESS doesn't fully guarantee algo will run (will fail if e.g. buffers are not correctly aligned);
    but if cublasLtMatmulAlgoCheck fails, the algo will not run.

    algo    algo configuration to check
    result  result structure to report algo runtime characteristics; algo field is never updated

    \retval     CUBLAS_STATUS_INVALID_VALUE  if matrix layout descriptors or operation descriptor don't match algo
                                            descriptor
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algo configuration or data type combination is not currently supported on
                                            given device
    \retval     CUBLAS_STATUS_ARCH_MISMATCH  if algo configuration cannot be run using the selected device
    \retval     CUBLAS_STATUS_SUCCESS        if check was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoCheck",
        fn (
            UnsafePointer[Context],
            UnsafePointer[Descriptor],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatmulAlgorithm],
            UnsafePointer[cublasLtMatmulHeuristicResult_t],
        ) -> Result,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        algo,
        result,
    )


@fieldwise_init
@register_passable("trivial")
struct Search:
    """Matmul heuristic search mode
    ."""

    var _value: Int32
    alias BEST_FIT = Self(0)
    """ask heuristics for best algo for given usecase.
    """
    alias LIMITED_BY_ALGO_ID = Self(1)
    """only try to find best config for preconfigured algo id.
    """
    alias RESERVED_02 = Self(2)
    """reserved for future use.
    """
    alias RESERVED_03 = Self(3)
    """reserved for future use.
    """
    alias RESERVED_04 = Self(4)
    """reserved for future use.
    """
    alias RESERVED_05 = Self(5)
    """reserved for future use.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.BEST_FIT:
            return "BEST_FIT"
        if self == Self.LIMITED_BY_ALGO_ID:
            return "LIMITED_BY_ALGO_ID"
        if self == Self.RESERVED_02:
            return "RESERVED_02"
        if self == Self.RESERVED_03:
            return "RESERVED_03"
        if self == Self.RESERVED_04:
            return "RESERVED_04"
        if self == Self.RESERVED_05:
            return "RESERVED_05"
        return abort[String]("invalid Search entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct ReductionScheme:
    """Reduction scheme for portions of the dot-product calculated in parallel (a. k. a. "split - K").
    ."""

    var _value: Int32
    alias NONE = ReductionScheme(0)
    """No reduction scheme, dot-product shall be performed in one sequence.
    """
    alias INPLACE = ReductionScheme(1)
    """Reduction is performed "in place" - using the output buffer (and output data type) and counters (in workspace) to
    guarantee the sequentiality.
    """
    alias COMPUTE_TYPE = ReductionScheme(2)
    """Intermediate results are stored in compute type in the workspace and reduced in a separate step.
    """
    alias OUTPUT_TYPE = ReductionScheme(3)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """
    alias MASK = ReductionScheme(4)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.NONE:
            return "NONE"
        if self == Self.INPLACE:
            return "INPLACE"
        if self == Self.COMPUTE_TYPE:
            return "COMPUTE_TYPE"
        if self == Self.OUTPUT_TYPE:
            return "OUTPUT_TYPE"
        if self == Self.MASK:
            return "MASK"
        return abort[String]("invalid ReductionScheme entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtLoggerSetCallback(
    callback: fn (
        Int16, UnsafePointer[Int8], UnsafePointer[NoneType]
    ) raises -> None
) -> Result:
    """Experimental: Logger callback setter.

    callback                     a user defined callback function to be called by the logger

    \retval     CUBLAS_STATUS_SUCCESS        if callback was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetCallback",
        fn (
            fn (
                Int16, UnsafePointer[Int8], UnsafePointer[NoneType]
            ) raises -> None
        ) -> Result,
    ]()(callback)


fn cublasLtGetProperty(type: Property, value: UnsafePointer[Int16]) -> Result:
    return _get_dylib_function[
        "cublasLtGetProperty",
        fn (Property, UnsafePointer[Int16]) -> Result,
    ]()(type, value)


fn cublasLtGetVersion() raises -> Int:
    return _get_dylib_function["cublasLtGetVersion", fn () -> Int]()()


fn cublasLtMatrixLayoutGetAttribute(
    mat_layout: UnsafePointer[MatrixLayout],
    attr: LayoutAttribute,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get matrix layout descriptor attribute.

    matLayout    The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutGetAttribute",
        fn (
            UnsafePointer[MatrixLayout],
            LayoutAttribute,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes, size_written)


@register_passable("trivial")
struct PreferenceOpaque:
    """Semi-opaque descriptor for cublasLtMatmulSelf() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@fieldwise_init
@register_passable("trivial")
struct cublasLtMatmulDescAttributes_t:
    """Matmul descriptor attributes to define details of the operation. ."""

    var _value: Int32
    alias CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = Self(0)
    """Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
    accumulator during matrix multiplication.

    int32_t.
    """
    alias CUBLASLT_MATMUL_DESC_SCALE_TYPE = Self(1)
    """Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
    typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
    D before being stored in memory.

    int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE.
    """
    alias CUBLASLT_MATMUL_DESC_POINTER_MODE = Self(2)
    """UnsafePointer mode of alpha and beta, see PointerMode. When DEVICE_VECTOR is in use,
    alpha/beta vector lengths must match number of output matrix rows.

    int32_t, default: HOST.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSA = Self(3)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSB = Self(4)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSC = Self(5)
    """Transform of matrix C, see cublasOperation_t.

    Currently only CUBLAS_OP_N is supported.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_FILL_MODE = Self(6)
    """Matrix fill mode, see cublasFillMode_t.

    int32_t, default: CUBLAS_FILL_MODE_FULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE = Self(7)
    """Epilogue function, see Epilogue.

    uint32_t, default: DEFAULT.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_POINTER = Self(8)
    """Bias or bias gradient vector pointer in the device memory.

    Bias case. See BIAS.
    For bias data type see CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE.

    Bias vector length must match matrix D rows count.

    Bias gradient case. See DRELU_BGRAD and DGELU_BGRAD.
    Bias gradient vector elements are the same type as the output elements
    (Ctype) with the exception of IMMA kernels (see above).

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Bias case: const void *, default: NULL
    Bias gradient case: void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = Self(9)
    """Batch stride for bias or bias gradient vector.

    Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's BATCH_COUNT > 1.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = Self(10)
    """UnsafePointer for epilogue auxiliary buffer.

    - Output vector for ReLu bit-mask in forward pass when RELU_AUX
     or RELU_AUX_BIAS epilogue is used.
    - Input vector for ReLu bit-mask in backward pass when
     DRELU_BGRAD epilogue is used.

    - Output of GELU input matrix in forward pass when
     GELU_AUX_BIAS epilogue is used.
    - Input of GELU input matrix for backward pass when
     DGELU_BGRAD epilogue is used.

    For aux data type see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE.

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.

    Forward pass: void *, default: NULL
    Backward pass: const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = Self(11)
    """Leading dimension for epilogue auxiliary buffer.

    - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128 and be no less than the number of rows in the output matrix.

    - GELU input matrix leading dimension in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8 and be no less than the number of rows in the output matrix.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = Self(12)
    """Batch stride for epilogue auxiliary buffer.

    - ReLu bit-mask matrix batch stride in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128.

    - GELU input matrix batch stride in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = Self(13)
    """Batch stride for alpha vector.

    Used together with ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
    BATCH_COUNT > 1. If ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesnt supported batched alpha vector.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = Self(14)
    """Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs
    when user expects a concurrent stream to be using some of the device resources.

    int32_t, default: 0 - use the number reported by the device.
    """
    alias CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = Self(15)
    """Device pointer to the scale factor value that converts data in matrix A to the compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = Self(16)
    """Device pointer to the scale factor value to convert data in matrix B to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = Self(17)
    """Device pointer to the scale factor value to convert data in matrix C to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = Self(18)
    """Device pointer to the scale factor value to convert data in matrix D to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = Self(19)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    output matrix.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = Self(20)
    """Type of the data to be stored to the memory pointed to by CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    If unset, the data type defaults to the type of elements of the output matrix with some exceptions, see details
    below.

    ReLu uses a bit-mask.

    GELU input matrix elements type is the same as the type of elements of
    the output matrix with some exceptions, see details below.

    For fp8 kernels with output type CUDA_R_8F_E4M3 the aux data type can be CUDA_R_8F_E4M3 or CUDA_R_16F with some
    restrictions.  See https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t for more details.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    int32_t based on cudaDataType, default: -1.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = Self(21)
    """Device pointer to the scaling factor value to convert results from compute type data range to storage
    data range in the auxiliary matrix that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data,
    scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = Self(22)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    buffer that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_FAST_ACCUM = Self(23)
    """Flag for managing fp8 fast accumulation mode.
    When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results
    will not periodically be promoted to a higher precision.

    int8_t, default: 0 - fast accumulation mode is disabled.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = Self(24)
    """Type of bias or bias gradient vector in the device memory.

    Bias case: see BIAS.

    Bias vector elements are the same type as the elements of output matrix (Dtype) with the following exceptions:
    - IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements
     are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
    - fp8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2, See
     https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul for details.

    int32_t based on cudaDataType, default: -1.
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = Self(25)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the row dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = Self(26)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the column dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = Self(27)
    """EXPERIMENTAL: UnsafePointer to a device array of input atomic counters consumed by a matmul.

    int32_t *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = Self(28)
    """EXPERIMENTAL: UnsafePointer to a device array of output atomic counters produced by a matmul.

    int32_t *, default: NULL.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
            return "CUBLASLT_MATMUL_DESC_COMPUTE_TYPE"
        if self == Self.CUBLASLT_MATMUL_DESC_SCALE_TYPE:
            return "CUBLASLT_MATMUL_DESC_SCALE_TYPE"
        if self == Self.CUBLASLT_MATMUL_DESC_POINTER_MODE:
            return "CUBLASLT_MATMUL_DESC_POINTER_MODE"
        if self == Self.CUBLASLT_MATMUL_DESC_TRANSA:
            return "CUBLASLT_MATMUL_DESC_TRANSA"
        if self == Self.CUBLASLT_MATMUL_DESC_TRANSB:
            return "CUBLASLT_MATMUL_DESC_TRANSB"
        if self == Self.CUBLASLT_MATMUL_DESC_TRANSC:
            return "CUBLASLT_MATMUL_DESC_TRANSC"
        if self == Self.CUBLASLT_MATMUL_DESC_FILL_MODE:
            return "CUBLASLT_MATMUL_DESC_FILL_MODE"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE"
        if self == Self.CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            return "CUBLASLT_MATMUL_DESC_BIAS_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
            return "CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE"
        if self == Self.CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE:
            return "CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE"
        if self == Self.CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
            return "CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET"
        if self == Self.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            return "CUBLASLT_MATMUL_DESC_A_SCALE_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            return "CUBLASLT_MATMUL_DESC_B_SCALE_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_C_SCALE_POINTER:
            return "CUBLASLT_MATMUL_DESC_C_SCALE_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
            return "CUBLASLT_MATMUL_DESC_D_SCALE_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_AMAX_D_POINTER:
            return "CUBLASLT_MATMUL_DESC_AMAX_D_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER:
            return "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_FAST_ACCUM:
            return "CUBLASLT_MATMUL_DESC_FAST_ACCUM"
        if self == Self.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            return "CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE"
        if self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS:
            return "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS"
        if self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS:
            return "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS"
        if self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER:
            return "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER"
        if self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER:
            return "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER"
        return abort[String]("invalid cublasLtMatmulDescAttributes_t entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtMatrixTransformDescInit_internal(
    transform_desc: UnsafePointer[Transform],
    size: Int,
    scale_type: DataType,
) -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescInit_internal",
        fn (UnsafePointer[Transform], Int, DataType) -> Result,
    ]()(transform_desc, size, scale_type)


fn cublasLtMatrixLayoutDestroy(
    mat_layout: UnsafePointer[MatrixLayout],
) -> Result:
    """Destroy matrix layout descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutDestroy",
        fn (UnsafePointer[MatrixLayout]) -> Result,
    ]()(mat_layout)


# Opaque descriptor for cublasLtMatmul() operation details
# .
alias cublasLtMatmulDesc_t = UnsafePointer[Descriptor]

# Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
# .
alias cublasLtMatmulPreference_t = UnsafePointer[PreferenceOpaque]


fn cublasLtMatmul(
    light_handle: UnsafePointer[Context],
    compute_desc: UnsafePointer[Descriptor],
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _adesc: UnsafePointer[MatrixLayout],
    _b: UnsafePointer[NoneType],
    _bdesc: UnsafePointer[MatrixLayout],
    beta: UnsafePointer[NoneType],
    _c: UnsafePointer[NoneType],
    _cdesc: UnsafePointer[MatrixLayout],
    _d: UnsafePointer[NoneType],
    _ddesc: UnsafePointer[MatrixLayout],
    algo: UnsafePointer[MatmulAlgorithm],
    workspace: UnsafePointer[NoneType],
    workspace_size_in_bytes: Int,
    stream: _CUstream_st,
) -> Result:
    """Execute matrix multiplication (D = alpha * op(A) * op(B) + beta * C).

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when workspaceSizeInBytes is less than workspace required by configured
                                               algo
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmul",
        fn (
            UnsafePointer[Context],
            UnsafePointer[Descriptor],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatmulAlgorithm],
            UnsafePointer[NoneType],
            Int,
            _CUstream_st,
        ) -> Result,
    ]()(
        light_handle,
        compute_desc,
        alpha,
        _a,
        _adesc,
        _b,
        _bdesc,
        beta,
        _c,
        _cdesc,
        _d,
        _ddesc,
        algo,
        workspace,
        workspace_size_in_bytes,
        stream,
    )


fn cublasLtMatrixTransformDescDestroy(
    transform_desc: UnsafePointer[Transform],
) -> Result:
    """Destroy matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescDestroy",
        fn (UnsafePointer[Transform]) -> Result,
    ]()(transform_desc)


fn cublasLtMatmulAlgoCapGetAttribute(
    algo: UnsafePointer[MatmulAlgorithm],
    attr: MatmulAlgorithmCapability,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get algo capability attribute.

    E.g. to get list of supported Tile IDs:
        Tile tiles[TILE_END];
        size_t num_tiles, size_written;
        if (cublasLtMatmulAlgoCapGetAttribute(algo, TILE_IDS, tiles, sizeof(tiles), size_written) ==
    CUBLAS_STATUS_SUCCESS) { num_tiles = size_written / sizeof(tiles[0]);
        }

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoCapGetAttribute",
        fn (
            UnsafePointer[MatmulAlgorithm],
            MatmulAlgorithmCapability,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulDescSetAttribute(
    matmul_desc: UnsafePointer[Descriptor],
    attr: cublasLtMatmulDescAttributes_t,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) -> Result:
    """Set matmul operation descriptor attribute.

    matmulDesc   The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescSetAttribute",
        fn (
            UnsafePointer[Descriptor],
            cublasLtMatmulDescAttributes_t,
            UnsafePointer[NoneType],
            Int,
        ) -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes)


fn cublasLtMatmulPreferenceSetAttribute(
    pref: UnsafePointer[PreferenceOpaque],
    attr: Preference,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) -> Result:
    """Set matmul heuristic search preference descriptor attribute.

    pref         The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceSetAttribute",
        fn (
            UnsafePointer[PreferenceOpaque],
            Preference,
            UnsafePointer[NoneType],
            Int,
        ) -> Result,
    ]()(pref, attr, buf, size_in_bytes)


# Experimental: Logger callback type.
# .
alias cublasLtLoggerCallback_t = fn (
    Int32, UnsafePointer[Int8], UnsafePointer[Int8]
) -> None


fn cublasLtMatrixLayoutInit_internal(
    mat_layout: UnsafePointer[MatrixLayout],
    size: Int,
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutInit_internal",
        fn (
            UnsafePointer[MatrixLayout],
            Int,
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) -> Result,
    ]()(mat_layout, size, type, rows, cols, ld)


@fieldwise_init
@register_passable("trivial")
struct Preference:
    """Algo search preference to fine tune the heuristic function. ."""

    var _value: Int32
    alias SEARCH_MODE = Self(0)
    """Search mode, see Search.

    uint32_t, default: BEST_FIT.
    """
    alias MAX_WORKSPACE_BYTES = Self(1)
    """Maximum allowed workspace size in bytes.

    uint64_t, default: 0 - no workspace allowed.
    """
    alias REDUCTION_SCHEME_MASK = Self(2)
    """Reduction scheme mask, see ReductionScheme. Filters heuristic result to only include algo configs that
    use one of the required modes.

    E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.

    uint32_t, default: MASK (allows all reduction schemes).
    """
    alias MIN_ALIGNMENT_A_BYTES = Self(3)
    """Minimum buffer alignment for matrix A (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias MIN_ALIGNMENT_B_BYTES = Self(4)
    """Minimum buffer alignment for matrix B (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias MIN_ALIGNMENT_C_BYTES = Self(5)
    """Minimum buffer alignment for matrix C (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias MIN_ALIGNMENT_D_BYTES = Self(6)
    """Minimum buffer alignment for matrix D (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias MAX_WAVES_COUNT = Self(7)
    """Maximum wave count.

    See cublasLtMatmulHeuristicResult_t::wavesCount.

    Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.

    float, default: 0.0f.
    """
    alias IMPL_MASK = Self(8)
    """Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
    algorithms that use the allowed implementations.

    uint64_t, default: uint64_t(-1) (allow everything).
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.SEARCH_MODE:
            return "SEARCH_MODE"
        if self == Self.MAX_WORKSPACE_BYTES:
            return "MAX_WORKSPACE_BYTES"
        if self == Self.REDUCTION_SCHEME_MASK:
            return "REDUCTION_SCHEME_MASK"
        if self == Self.MIN_ALIGNMENT_A_BYTES:
            return "MIN_ALIGNMENT_A_BYTES"
        if self == Self.MIN_ALIGNMENT_B_BYTES:
            return "MIN_ALIGNMENT_B_BYTES"
        if self == Self.MIN_ALIGNMENT_C_BYTES:
            return "MIN_ALIGNMENT_C_BYTES"
        if self == Self.MIN_ALIGNMENT_D_BYTES:
            return "MIN_ALIGNMENT_D_BYTES"
        if self == Self.MAX_WAVES_COUNT:
            return "MAX_WAVES_COUNT"
        if self == Self.IMPL_MASK:
            return "IMPL_MASK"
        return abort[String]("invalid Preference entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


@register_passable("trivial")
struct MatmulAlgorithm:
    """Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes).

    This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
    on selecting the right configuration again.
    """

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]

    fn __init__(out self):
        self.data = StaticTuple[UInt64, 8](0)


alias cublasLtNumericalImplFlags_t = UInt64


@fieldwise_init
@register_passable("trivial")
struct AlgorithmConfig:
    """Algo Configuration Attributes that can be set according to the Algo capabilities
    ."""

    var _value: Int32
    alias ID = Self(0)
    """algorithm index, see cublasLtMatmulAlgoGetIds()

    readonly, set by cublasLtMatmulAlgoInit()
    int32_t.
    """
    alias TILE_ID = Self(1)
    """tile id, see Tile

    uint32_t, default: TILE_UNDEFINED.
    """
    alias SPLITK_NUM = Self(2)
    """Number of K splits. If the number of K splits is greater than one, SPLITK_NUM parts
    of matrix multiplication will be computed in parallel. The results will be accumulated
    according to REDUCTION_SCHEME

    int32_t, default: 1.
    """
    alias REDUCTION_SCHEME = Self(3)
    """reduction scheme, see ReductionScheme

    uint32_t, default: NONE.
    """
    alias CTA_SWIZZLING = Self(4)
    """cta swizzling, change mapping from CUDA grid coordinates to parts of the matrices

    possible values: 0, 1, other values reserved

    uint32_t, default: 0.
    """
    alias CUSTOM_OPTION = Self(5)
    """custom option, each algorithm can support some custom options that don't fit description of the other config
    attributes, see CUSTOM_OPTION_MAX to get accepted range for any specific case

    uint32_t, default: 0.
    """
    alias STAGES_ID = Self(6)
    """stages id, see Stages

    uint32_t, default: STAGES_UNDEFINED.
    """
    alias INNER_SHAPE_ID = Self(7)
    """inner shape id, see InnerShape

    uint16_t, default: 0 (UNDEFINED).
    """
    alias CLUSTER_SHAPE_ID = Self(8)
    """Thread Block Cluster shape id, see ClusterShape. Defines cluster size to use.

    uint16_t, default: 0 (SHAPE_AUTO).
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.ID:
            return "ID"
        if self == Self.TILE_ID:
            return "TILE_ID"
        if self == Self.SPLITK_NUM:
            return "SPLITK_NUM"
        if self == Self.REDUCTION_SCHEME:
            return "REDUCTION_SCHEME"
        if self == Self.CTA_SWIZZLING:
            return "CTA_SWIZZLING"
        if self == Self.CUSTOM_OPTION:
            return "CUSTOM_OPTION"
        if self == Self.STAGES_ID:
            return "STAGES_ID"
        if self == Self.INNER_SHAPE_ID:
            return "INNER_SHAPE_ID"
        if self == Self.CLUSTER_SHAPE_ID:
            return "CLUSTER_SHAPE_ID"
        return abort[String]("invalid AlgorithmConfig entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtMatmulPreferenceDestroy(
    pref: UnsafePointer[PreferenceOpaque],
) -> Result:
    """Destroy matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceDestroy",
        fn (UnsafePointer[PreferenceOpaque]) -> Result,
    ]()(pref)


fn cublasLtMatmulAlgoGetHeuristic(
    light_handle: UnsafePointer[Context],
    operation_desc: UnsafePointer[Descriptor],
    _adesc: UnsafePointer[MatrixLayout],
    _bdesc: UnsafePointer[MatrixLayout],
    _cdesc: UnsafePointer[MatrixLayout],
    _ddesc: UnsafePointer[MatrixLayout],
    preference: UnsafePointer[PreferenceOpaque],
    requested_algo_count: Int,
    heuristic_results_array: UnsafePointer[cublasLtMatmulHeuristicResult_t],
    return_algo_count: UnsafePointer[Int],
) -> Result:
    """Query cublasLt heuristic for algorithm appropriate for given use case.

        lightHandle            UnsafePointer to the allocated cuBLASLt handle for the cuBLASLt
                                          context. See cublasLtHandle_t.
        operationDesc          Handle to the matrix multiplication descriptor.
        Adesc                  Handle to the layout descriptors for matrix A.
        Bdesc                  Handle to the layout descriptors for matrix B.
        Cdesc                  Handle to the layout descriptors for matrix C.
        Ddesc                  Handle to the layout descriptors for matrix D.
        preference             UnsafePointer to the structure holding the heuristic search
                                          preferences descriptor. See cublasLtMatrixLayout_t.
        requestedAlgoCount     Size of heuristicResultsArray (in elements) and requested
                                          maximum number of algorithms to return.
        returnAlgoCount        The number of heuristicResultsArray elements written.

    \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal to zero
    \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function available for current configuration
    \retval  CUBLAS_STATUS_SUCCESS         if query was successful, inspect
                                          heuristicResultsArray[0 to (returnAlgoCount - 1)].state
                                          for detail status of results
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoGetHeuristic",
        fn (
            UnsafePointer[Context],
            UnsafePointer[Descriptor],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[MatrixLayout],
            UnsafePointer[PreferenceOpaque],
            Int16,
            UnsafePointer[cublasLtMatmulHeuristicResult_t],
            UnsafePointer[Int],
        ) -> Result,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        preference,
        requested_algo_count,
        heuristic_results_array,
        return_algo_count,
    )


@fieldwise_init
@register_passable("trivial")
struct InnerShape:
    """Inner size of the kernel.

    Represents various aspects of internal kernel design, that don't impact CUDA grid size but may have other more subtle
    effects.
    """

    var _value: Int32
    alias UNDEFINED = Self(0)
    alias MMA884 = Self(1)
    alias MMA1684 = Self(2)
    alias MMA1688 = Self(3)
    alias MMA16816 = Self(4)
    alias END = Self(5)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.UNDEFINED:
            return "UNDEFINED"
        if self == Self.MMA884:
            return "MMA884"
        if self == Self.MMA1684:
            return "MMA1684"
        if self == Self.MMA1688:
            return "MMA1688"
        if self == Self.MMA16816:
            return "MMA16816"
        if self == Self.END:
            return "END"
        return abort[String]("invalid InnerShape entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct LayoutAttribute:
    """Attributes of memory layout ."""

    var _value: Int32
    alias TYPE = Self(0)
    """Data type, see cudaDataType.

    uint32_t.
    """
    alias ORDER = Self(1)
    """Memory order of the data, see Order.

    int32_t, default: COL.
    """
    alias ROWS = Self(2)
    """Number of rows.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    alias COLS = Self(3)
    """Number of columns.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    alias LD = Self(4)
    """Matrix leading dimension.

    For COL this is stride (in elements) of matrix column, for more details and documentation for
    other memory orders see documentation for Order values.

    Currently only non-negative values are supported, must be large enough so that matrix memory locations are not
    overlapping (e.g. greater or equal to ROWS in case of COL).

    int64_t;.
    """
    alias BATCH_COUNT = Self(5)
    """Number of matmul operations to perform in the batch.

    See also STRIDED_BATCH_SUPPORT

    int32_t, default: 1.
    """
    alias STRIDED_BATCH_OFFSET = Self(6)
    """Stride (in elements) to the next matrix for strided batch operation.

    When matrix type is planar-complex (PLANE_OFFSET != 0), batch stride
    is interpreted by cublasLtMatmul() in number of real valued sub-elements. E.g. for data of type CUDA_C_16F,
    offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices
    is a 2B (16bit) floating point type).

    NOTE: A bug in cublasLtMatrixTransform() causes it to interpret the batch stride for a planar-complex matrix
    as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride
    value 256 when calling cublasLtMatrixTransform() (each complex element is 4B with real and imaginary values 2B
    each). This behavior is expected to be corrected in the next major cuBLAS version.

    int64_t, default: 0.
    """
    alias PLANE_OFFSET = Self(7)
    """Stride (in bytes) to the imaginary plane for planar complex layout.

    int64_t, default: 0 - 0 means that layout is regular (real and imaginary parts of complex numbers are interleaved
    in memory in each element).
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.TYPE:
            return "TYPE"
        if self == Self.ORDER:
            return "ORDER"
        if self == Self.ROWS:
            return "ROWS"
        if self == Self.COLS:
            return "COLS"
        if self == Self.LD:
            return "LD"
        if self == Self.BATCH_COUNT:
            return "BATCH_COUNT"
        if self == Self.STRIDED_BATCH_OFFSET:
            return "STRIDED_BATCH_OFFSET"
        if self == Self.PLANE_OFFSET:
            return "PLANE_OFFSET"
        return abort[String]("invalid LayoutAttribute entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtDestroy(light_handle: UnsafePointer[Context]) -> Result:
    return _get_dylib_function[
        "cublasLtDestroy", fn (UnsafePointer[Context]) -> Result
    ]()(light_handle)


fn cublasLtGetCudartVersion() raises -> Int:
    return _get_dylib_function["cublasLtGetCudartVersion", fn () -> Int]()()


fn cublasLtMatmulAlgoConfigGetAttribute(
    algo: UnsafePointer[MatmulAlgorithm],
    attr: AlgorithmConfig,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get algo configuration attribute.

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoConfigGetAttribute",
        fn (
            UnsafePointer[MatmulAlgorithm],
            AlgorithmConfig,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


fn cublasLtLoggerForceDisable() -> Result:
    """Experimental: Disable logging for the entire session.

    \retval     CUBLAS_STATUS_SUCCESS        if disabled logging
    ."""
    return _get_dylib_function[
        "cublasLtLoggerForceDisable", fn () -> Result
    ]()()


fn cublasLtHeuristicsCacheGetCapacity(
    capacity: UnsafePointer[Int],
) -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheGetCapacity",
        fn (UnsafePointer[Int]) -> Result,
    ]()(capacity)


fn cublasLtDisableCpuInstructionsSetMask(mask: Int16) raises -> Int16:
    """Restricts usage of CPU instructions (ISA) specified by the flags in the mask.

    Flags can be combined with bitwise OR(|) operator. Supported flags:
    - 0x1 -- x86-64 AVX512 ISA

    Default mask: 0 (any applicable ISA is allowed).

    The function returns the previous value of the mask.
    The function takes precedence over the environment variable CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK.
    ."""
    return _get_dylib_function[
        "cublasLtDisableCpuInstructionsSetMask", fn (Int16) raises -> Int16
    ]()(mask)


fn cublasLtLoggerSetLevel(level: Int16) -> Result:
    """Experimental: Log level setter.

    level                        log level, should be one of the following:
                                            0. Off
                                            1. Errors
                                            2. Performance Trace
                                            3. Performance Hints
                                            4. Heuristics Trace
                                            5. API Trace

    \retval     CUBLAS_STATUS_INVALID_VALUE  if log level is not one of the above levels

    \retval     CUBLAS_STATUS_SUCCESS        if log level was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetLevel", fn (Int16) -> Result
    ]()(level)


@fieldwise_init
@register_passable("trivial")
struct Stages:
    """Size and number of stages in which elements are read into shared memory.

    General order of stages IDs is sorted by stage size first and by number of stages second.
    ."""

    var _value: Int32
    alias STAGES_UNDEFINED = Self(0)
    alias STAGES_16x1 = Self(1)
    alias STAGES_16x2 = Self(2)
    alias STAGES_16x3 = Self(3)
    alias STAGES_16x4 = Self(4)
    alias STAGES_16x5 = Self(5)
    alias STAGES_16x6 = Self(6)
    alias STAGES_32x1 = Self(7)
    alias STAGES_32x2 = Self(8)
    alias STAGES_32x3 = Self(9)
    alias STAGES_32x4 = Self(10)
    alias STAGES_32x5 = Self(11)
    alias STAGES_32x6 = Self(12)
    alias STAGES_64x1 = Self(13)
    alias STAGES_64x2 = Self(14)
    alias STAGES_64x3 = Self(15)
    alias STAGES_64x4 = Self(16)
    alias STAGES_64x5 = Self(17)
    alias STAGES_64x6 = Self(18)
    alias STAGES_128x1 = Self(19)
    alias STAGES_128x2 = Self(20)
    alias STAGES_128x3 = Self(21)
    alias STAGES_128x4 = Self(22)
    alias STAGES_128x5 = Self(23)
    alias STAGES_128x6 = Self(24)
    alias STAGES_32x10 = Self(25)
    alias STAGES_8x4 = Self(26)
    alias STAGES_16x10 = Self(27)
    alias STAGES_8x5 = Self(28)
    alias STAGES_8x3 = Self(29)
    alias STAGES_8xAUTO = Self(30)
    alias STAGES_16xAUTO = Self(31)
    alias STAGES_32xAUTO = Self(32)
    alias STAGES_64xAUTO = Self(33)
    alias STAGES_128xAUTO = Self(34)
    alias STAGES_END = Self(35)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.STAGES_UNDEFINED:
            return "STAGES_UNDEFINED"
        if self == Self.STAGES_16x1:
            return "STAGES_16x1"
        if self == Self.STAGES_16x2:
            return "STAGES_16x2"
        if self == Self.STAGES_16x3:
            return "STAGES_16x3"
        if self == Self.STAGES_16x4:
            return "STAGES_16x4"
        if self == Self.STAGES_16x5:
            return "STAGES_16x5"
        if self == Self.STAGES_16x6:
            return "STAGES_16x6"
        if self == Self.STAGES_32x1:
            return "STAGES_32x1"
        if self == Self.STAGES_32x2:
            return "STAGES_32x2"
        if self == Self.STAGES_32x3:
            return "STAGES_32x3"
        if self == Self.STAGES_32x4:
            return "STAGES_32x4"
        if self == Self.STAGES_32x5:
            return "STAGES_32x5"
        if self == Self.STAGES_32x6:
            return "STAGES_32x6"
        if self == Self.STAGES_64x1:
            return "STAGES_64x1"
        if self == Self.STAGES_64x2:
            return "STAGES_64x2"
        if self == Self.STAGES_64x3:
            return "STAGES_64x3"
        if self == Self.STAGES_64x4:
            return "STAGES_64x4"
        if self == Self.STAGES_64x5:
            return "STAGES_64x5"
        if self == Self.STAGES_64x6:
            return "STAGES_64x6"
        if self == Self.STAGES_128x1:
            return "STAGES_128x1"
        if self == Self.STAGES_128x2:
            return "STAGES_128x2"
        if self == Self.STAGES_128x3:
            return "STAGES_128x3"
        if self == Self.STAGES_128x4:
            return "STAGES_128x4"
        if self == Self.STAGES_128x5:
            return "STAGES_128x5"
        if self == Self.STAGES_128x6:
            return "STAGES_128x6"
        if self == Self.STAGES_32x10:
            return "STAGES_32x10"
        if self == Self.STAGES_8x4:
            return "STAGES_8x4"
        if self == Self.STAGES_16x10:
            return "STAGES_16x10"
        if self == Self.STAGES_8x5:
            return "STAGES_8x5"
        if self == Self.STAGES_8x3:
            return "STAGES_8x3"
        if self == Self.STAGES_8xAUTO:
            return "STAGES_8xAUTO"
        if self == Self.STAGES_16xAUTO:
            return "STAGES_16xAUTO"
        if self == Self.STAGES_32xAUTO:
            return "STAGES_32xAUTO"
        if self == Self.STAGES_64xAUTO:
            return "STAGES_64xAUTO"
        if self == Self.STAGES_128xAUTO:
            return "STAGES_128xAUTO"
        if self == Self.STAGES_END:
            return "STAGES_END"
        return abort[String]("invalid Stages entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtMatmulDescDestroy(
    matmul_desc: UnsafePointer[Descriptor],
) -> Result:
    """Destroy matmul operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescDestroy",
        fn (UnsafePointer[Descriptor]) -> Result,
    ]()(matmul_desc)


fn cublasLtMatrixTransformDescSetAttribute(
    transform_desc: UnsafePointer[Transform],
    attr: TransformDescriptor,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
) -> Result:
    """Set matrix transform operation descriptor attribute.

    transformDesc  The descriptor
    attr           The attribute
    buf            memory address containing the new value
    sizeInBytes    size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescSetAttribute",
        fn (
            UnsafePointer[Transform],
            TransformDescriptor,
            UnsafePointer[NoneType],
            Int,
        ) -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes)


fn cublasLtMatmulPreferenceGetAttribute(
    pref: UnsafePointer[PreferenceOpaque],
    attr: Preference,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get matmul heuristic search preference descriptor attribute.

    pref         The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceGetAttribute",
        fn (
            UnsafePointer[PreferenceOpaque],
            Preference,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(pref, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulAlgoInit(
    light_handle: UnsafePointer[Context],
    compute_type: ComputeType,
    scale_type: DataType,
    _atype: DataType,
    _btype: DataType,
    _ctype: DataType,
    _dtype: DataType,
    algo_id: Int16,
    algo: UnsafePointer[MatmulAlgorithm],
) -> Result:
    """Initialize algo structure.

    \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside of recognized range
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not supported for given combination of data types
    \retval     CUBLAS_STATUS_SUCCESS        if the structure was successfully initialized
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoInit",
        fn (
            UnsafePointer[Context],
            ComputeType,
            DataType,
            DataType,
            DataType,
            DataType,
            DataType,
            Int16,
            UnsafePointer[MatmulAlgorithm],
        ) -> Result,
    ]()(
        light_handle,
        compute_type,
        scale_type,
        _atype,
        _btype,
        _ctype,
        _dtype,
        algo_id,
        algo,
    )


@fieldwise_init
@register_passable("trivial")
struct Epilogue:
    """Postprocessing options for the epilogue
    ."""

    var _value: Int32
    alias DEFAULT = Self(0)
    """No special postprocessing, just scale and quantize results if necessary.
    """
    alias RELU = Self(1)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
    """
    alias RELU_AUX = Self(2)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias BIAS = Self(3)
    """Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed
    (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final
    postprocessing.
    """
    alias RELU_BIAS = Self(4)
    """ReLu and Bias, apply Bias and then ReLu transform.
    """
    alias RELU_AUX_BIAS = Self(5)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias DRELU = Self(6)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias DRELU_BGRAD = Self(7)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias GELU = Self(8)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    """
    alias GELU_AUX = Self(9)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias GELU_BIAS = Self(10)
    """GELU and Bias, apply Bias and then GELU transform.
    """
    alias GELU_AUX_BIAS = Self(11)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias DGELU = Self(12)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias DGELU_BGRAD = Self(13)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias BGRADA = Self(14)
    """Bias gradient based on the input matrix A.

    The bias size corresponds to the number of rows of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """
    alias BGRADB = Self(15)
    """Bias gradient based on the input matrix B.

    The bias size corresponds to the number of columns of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.DEFAULT:
            return "DEFAULT"
        if self == Self.RELU:
            return "RELU"
        if self == Self.RELU_AUX:
            return "RELU_AUX"
        if self == Self.BIAS:
            return "BIAS"
        if self == Self.RELU_BIAS:
            return "RELU_BIAS"
        if self == Self.RELU_AUX_BIAS:
            return "RELU_AUX_BIAS"
        if self == Self.DRELU:
            return "DRELU"
        if self == Self.DRELU_BGRAD:
            return "DRELU_BGRAD"
        if self == Self.GELU:
            return "GELU"
        if self == Self.GELU_AUX:
            return "GELU_AUX"
        if self == Self.GELU_BIAS:
            return "GELU_BIAS"
        if self == Self.GELU_AUX_BIAS:
            return "GELU_AUX_BIAS"
        if self == Self.DGELU:
            return "DGELU"
        if self == Self.DGELU_BGRAD:
            return "DGELU_BGRAD"
        if self == Self.BGRADA:
            return "BGRADA"
        if self == Self.BGRADB:
            return "BGRADB"
        return abort[String]("invalid Epilogue entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


@register_passable("trivial")
struct Descriptor:
    """Semi-opaque descriptor for cublasLtMatmul() operation details
    ."""

    var data: StaticTuple[UInt64, 32]


fn cublasLtMatrixLayoutCreate(
    mat_layout: UnsafePointer[UnsafePointer[MatrixLayout]],
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) -> Result:
    """Create new matrix layout descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutCreate",
        fn (
            UnsafePointer[UnsafePointer[MatrixLayout]],
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) -> Result,
    ]()(mat_layout, type, rows, cols, ld)


@fieldwise_init
@register_passable("trivial")
struct PointerModeMask:
    """Mask to define pointer mode capability ."""

    var _value: Int32
    alias HOST = Self(0)
    """see HOST."""
    alias DEVICE = Self(1)
    """see DEVICE."""
    alias DEVICE_VECTOR = Self(2)
    """see DEVICE_VECTOR."""
    alias ALPHA_DEVICE_VECTOR_BETA_ZERO = Self(3)
    """see ALPHA_DEVICE_VECTOR_BETA_ZERO."""
    alias ALPHA_DEVICE_VECTOR_BETA_HOST = Self(4)
    """see ALPHA_DEVICE_VECTOR_BETA_HOST."""

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.HOST:
            return "HOST"
        if self == Self.DEVICE:
            return "DEVICE"
        if self == Self.DEVICE_VECTOR:
            return "DEVICE_VECTOR"
        if self == Self.ALPHA_DEVICE_VECTOR_BETA_ZERO:
            return "ALPHA_DEVICE_VECTOR_BETA_ZERO"
        if self == Self.ALPHA_DEVICE_VECTOR_BETA_HOST:
            return "ALPHA_DEVICE_VECTOR_BETA_HOST"
        return abort[String]("invalid PointerModeMask entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


@register_passable("trivial")
struct MatrixLayout:
    """Semi-opaque descriptor for matrix memory layout
    ."""

    var data: StaticTuple[UInt64, 8]


fn cublasLtMatmulDescCreate(
    matmul_desc: UnsafePointer[UnsafePointer[Descriptor]],
    compute_type: ComputeType,
    scale_type: DataType,
) -> Result:
    """Create new matmul operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescCreate",
        fn (
            UnsafePointer[UnsafePointer[Descriptor]],
            ComputeType,
            DataType,
        ) -> Result,
    ]()(matmul_desc, compute_type, scale_type)


@fieldwise_init
@register_passable("trivial")
struct Tile:
    """Tile size (in C/D matrix Rows x Cols).

    General order of tile IDs is sorted by size first and by first dimension second.
    ."""

    var _value: Int32
    alias TILE_UNDEFINED = Self(0)
    alias TILE_8x8 = Self(1)
    alias TILE_8x16 = Self(2)
    alias TILE_16x8 = Self(3)
    alias TILE_8x32 = Self(4)
    alias TILE_16x16 = Self(5)
    alias TILE_32x8 = Self(6)
    alias TILE_8x64 = Self(7)
    alias TILE_16x32 = Self(8)
    alias TILE_32x16 = Self(9)
    alias TILE_64x8 = Self(10)
    alias TILE_32x32 = Self(11)
    alias TILE_32x64 = Self(12)
    alias TILE_64x32 = Self(13)
    alias TILE_32x128 = Self(14)
    alias TILE_64x64 = Self(15)
    alias TILE_128x32 = Self(16)
    alias TILE_64x128 = Self(17)
    alias TILE_128x64 = Self(18)
    alias TILE_64x256 = Self(19)
    alias TILE_128x128 = Self(20)
    alias TILE_256x64 = Self(21)
    alias TILE_64x512 = Self(22)
    alias TILE_128x256 = Self(23)
    alias TILE_256x128 = Self(24)
    alias TILE_512x64 = Self(25)
    alias TILE_64x96 = Self(26)
    alias TILE_96x64 = Self(27)
    alias TILE_96x128 = Self(28)
    alias TILE_128x160 = Self(29)
    alias TILE_160x128 = Self(30)
    alias TILE_192x128 = Self(31)
    alias TILE_128x192 = Self(32)
    alias TILE_128x96 = Self(33)
    alias TILE_32x256 = Self(34)
    alias TILE_256x32 = Self(35)
    alias TILE_END = Self(36)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.TILE_UNDEFINED:
            return "TILE_UNDEFINED"
        if self == Self.TILE_8x8:
            return "TILE_8x8"
        if self == Self.TILE_8x16:
            return "TILE_8x16"
        if self == Self.TILE_16x8:
            return "TILE_16x8"
        if self == Self.TILE_8x32:
            return "TILE_8x32"
        if self == Self.TILE_16x16:
            return "TILE_16x16"
        if self == Self.TILE_32x8:
            return "TILE_32x8"
        if self == Self.TILE_8x64:
            return "TILE_8x64"
        if self == Self.TILE_16x32:
            return "TILE_16x32"
        if self == Self.TILE_32x16:
            return "TILE_32x16"
        if self == Self.TILE_64x8:
            return "TILE_64x8"
        if self == Self.TILE_32x32:
            return "TILE_32x32"
        if self == Self.TILE_32x64:
            return "TILE_32x64"
        if self == Self.TILE_64x32:
            return "TILE_64x32"
        if self == Self.TILE_32x128:
            return "TILE_32x128"
        if self == Self.TILE_64x64:
            return "TILE_64x64"
        if self == Self.TILE_128x32:
            return "TILE_128x32"
        if self == Self.TILE_64x128:
            return "TILE_64x128"
        if self == Self.TILE_128x64:
            return "TILE_128x64"
        if self == Self.TILE_64x256:
            return "TILE_64x256"
        if self == Self.TILE_128x128:
            return "TILE_128x128"
        if self == Self.TILE_256x64:
            return "TILE_256x64"
        if self == Self.TILE_64x512:
            return "TILE_64x512"
        if self == Self.TILE_128x256:
            return "TILE_128x256"
        if self == Self.TILE_256x128:
            return "TILE_256x128"
        if self == Self.TILE_512x64:
            return "TILE_512x64"
        if self == Self.TILE_64x96:
            return "TILE_64x96"
        if self == Self.TILE_96x64:
            return "TILE_96x64"
        if self == Self.TILE_96x128:
            return "TILE_96x128"
        if self == Self.TILE_128x160:
            return "TILE_128x160"
        if self == Self.TILE_160x128:
            return "TILE_160x128"
        if self == Self.TILE_192x128:
            return "TILE_192x128"
        if self == Self.TILE_128x192:
            return "TILE_128x192"
        if self == Self.TILE_128x96:
            return "TILE_128x96"
        if self == Self.TILE_32x256:
            return "TILE_32x256"
        if self == Self.TILE_256x32:
            return "TILE_256x32"
        if self == Self.TILE_END:
            return "TILE_END"
        return abort[String]("invalid Tile entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


fn cublasLtGetStatusName(status: Result) raises -> UnsafePointer[Int8]:
    return _get_dylib_function[
        "cublasLtGetStatusName", fn (Result) raises -> UnsafePointer[Int8]
    ]()(status)


fn cublasLtMatmulPreferenceCreate(
    pref: UnsafePointer[UnsafePointer[PreferenceOpaque]],
) -> Result:
    """Create new matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceCreate",
        fn (UnsafePointer[UnsafePointer[PreferenceOpaque]],) -> Result,
    ]()(pref)


@register_passable("trivial")
struct cublasLtMatmulHeuristicResult_t:
    """Results structure used by cublasLtMatmulGetAlgo.

    Holds returned configured algo descriptor and its runtime properties.
    ."""

    # Matmul algorithm descriptor.
    #
    # Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
    # LIMITED_BY_ALGO_ID
    # .
    var algo: MatmulAlgorithm
    # Actual size of workspace memory required.
    # .
    var workspaceSize: Int
    # Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set to
    # CUBLAS_STATUS_SUCCESS.
    # .
    var state: Result
    # Waves count - a device utilization metric.
    #
    # wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
    # .
    var wavesCount: Float32
    var reserved: StaticTuple[Int32, 4]

    fn __init__(out self):
        self.algo = MatmulAlgorithm()
        self.workspaceSize = 0
        self.state = Result.NOT_INITIALIZED
        self.wavesCount = 0.0
        self.reserved = StaticTuple[Int32, 4](0)


fn cublasLtLoggerSetFile(file: UnsafePointer[NoneType]) -> Result:
    """Experimental: Log file setter.

    file                         an open file with write permissions

    \retval     CUBLAS_STATUS_SUCCESS        if log file was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetFile", fn (UnsafePointer[NoneType]) -> Result
    ]()(file)


fn cublasLtLoggerOpenFile(log_file: UnsafePointer[Int8]) -> Result:
    """Experimental: Open log file.

    logFile                      log file path. if the log file does not exist, it will be created

    \retval     CUBLAS_STATUS_SUCCESS        if log file was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerOpenFile", fn (UnsafePointer[Int8]) -> Result
    ]()(log_file)


fn cublasLtMatrixTransform(
    light_handle: UnsafePointer[Context],
    transform_desc: UnsafePointer[Transform],
    alpha: UnsafePointer[NoneType],
    _a: UnsafePointer[NoneType],
    _adesc: UnsafePointer[MatrixLayout],
    beta: UnsafePointer[NoneType],
    _b: UnsafePointer[NoneType],
    _bdesc: UnsafePointer[MatrixLayout],
    _c: UnsafePointer[NoneType],
    _cdesc: UnsafePointer[MatrixLayout],
    stream: _CUstream_st,
) -> Result:
    """Matrix layout conversion helper (C = alpha * op(A) + beta * op(B)).

    Can be used to change memory order of data or to scale and shift the values.

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when A is not NULL, but Adesc is NULL
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransform",
        fn (
            UnsafePointer[Context],
            UnsafePointer[Transform],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[NoneType],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            UnsafePointer[NoneType],
            UnsafePointer[MatrixLayout],
            _CUstream_st,
        ) -> Result,
    ]()(
        light_handle,
        transform_desc,
        alpha,
        _a,
        _adesc,
        beta,
        _b,
        _bdesc,
        _c,
        _cdesc,
        stream,
    )


fn cublasLtLoggerSetMask(mask: Int16) -> Result:
    """Experimental: Log mask setter.

    mask                         log mask, should be a combination of the following masks:
                                            0.  Off
                                            1.  Errors
                                            2.  Performance Trace
                                            4.  Performance Hints
                                            8.  Heuristics Trace
                                            16. API Trace

    \retval     CUBLAS_STATUS_SUCCESS        if log mask was set successfully
    ."""
    return _get_dylib_function["cublasLtLoggerSetMask", fn (Int16) -> Result]()(
        mask
    )


# Opaque structure holding CUBLASLT context
# .
alias cublasLtHandle_t = UnsafePointer[Context]


fn cublasLtMatrixTransformDescGetAttribute(
    transform_desc: UnsafePointer[Transform],
    attr: TransformDescriptor,
    buf: UnsafePointer[NoneType],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int],
) -> Result:
    """Get matrix transform operation descriptor attribute.

    transformDesc  The descriptor
    attr           The attribute
    buf            memory address containing the new value
    sizeInBytes    size of buf buffer for verification (in bytes)
    sizeWritten    only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number
    of bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescGetAttribute",
        fn (
            UnsafePointer[Transform],
            TransformDescriptor,
            UnsafePointer[NoneType],
            Int,
            UnsafePointer[Int],
        ) -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulDescInit_internal(
    matmul_desc: UnsafePointer[Descriptor],
    size: Int,
    compute_type: ComputeType,
    scale_type: DataType,
) -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescInit_internal",
        fn (
            UnsafePointer[Descriptor],
            Int,
            ComputeType,
            DataType,
        ) -> Result,
    ]()(matmul_desc, size, compute_type, scale_type)


fn cublasLtMatmulPreferenceInit_internal(
    pref: UnsafePointer[PreferenceOpaque], size: Int
) -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceInit_internal",
        fn (UnsafePointer[PreferenceOpaque], Int) -> Result,
    ]()(pref, size)


@register_passable("trivial")
struct Transform:
    """Semi-opaque descriptor for cublasLtMatrixTransform() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@fieldwise_init
@register_passable("trivial")
struct TransformDescriptor:
    """Matrix transform descriptor attributes to define details of the operation.
    ."""

    var _value: Int32
    alias SCALE_TYPE = TransformDescriptor(0)
    """Scale type, see cudaDataType. Inputs are converted to scale type for scaling and summation and results are then
    converted to output type to store in memory.

    int32_t.
    """
    alias POINTER_MODE = TransformDescriptor(1)
    """UnsafePointer mode of alpha and beta, see PointerMode.

    int32_t, default: HOST.
    """
    alias TRANSA = TransformDescriptor(2)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias TRANSB = TransformDescriptor(3)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) raises -> String:
        if self == Self.SCALE_TYPE:
            return "SCALE_TYPE"
        if self == Self.POINTER_MODE:
            return "POINTER_MODE"
        if self == Self.TRANSA:
            return "TRANSA"
        if self == Self.TRANSB:
            return "TRANSB"
        return abort[String]("invalid TransformDescriptor entry")

    fn __int__(self) raises -> Int:
        return Int(self._value)


# fn cublasLtMatmulAlgoGetIds(
#     light_handle: UnsafePointer[Context],
#     compute_type: ComputeType,
#     scale_type: DataType,
#     _atype: DataType,
#     _btype: DataType,
#     _ctype: DataType,
#     _dtype: DataType,
#     requested_algo_count: Int16,
#     algo_ids_array: UNKNOWN,
#     return_algo_count: UnsafePointer[Int16],
# ) -> Result:
#     """Routine to get all algo IDs that can potentially run

#     int              requestedAlgoCount requested number of algos (must be less or equal to size of algoIdsA
#     (in elements)) algoIdsA         array to write algoIds to returnAlgoCount  number of algoIds
#     actually written

#     \retval     CUBLAS_STATUS_INVALID_VALUE  if requestedAlgoCount is less or equal to zero
#     \retval     CUBLAS_STATUS_SUCCESS        if query was successful, inspect returnAlgoCount to get actual number of IDs
#                                             available
#     ."""
#     return _get_dylib_function[
#         "cublasLtMatmulAlgoGetIds",
#         fn (
#             UnsafePointer[Context],
#             ComputeType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             Int16,
#             UNKNOWN,
#             UnsafePointer[Int16],
#         ) -> Result,
#     ]()(
#         light_handle,
#         compute_type,
#         scale_type,
#         _atype,
#         _btype,
#         _ctype,
#         _dtype,
#         requested_algo_count,
#         algo_ids_array,
#         return_algo_count,
#     )
