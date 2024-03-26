# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .result import Result
from .dtype import DataType, Property
from .cublas import ComputeType
from gpu.host import Stream
from collections import List
from os import abort
from pathlib import Path
from sys.ffi import DLHandle
from sys.ffi import _get_dylib_function as _ffi_get_dylib_function

from memory.unsafe import Pointer

alias cublasLtContext = NoneType

# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#

alias CUDA_CUBLASLT_LIBRARY_PATH = "/usr/local/cuda/lib64/libcublasLt.so"


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    if not Path(CUDA_CUBLASLT_LIBRARY_PATH).exists():
        return abort[Pointer[NoneType]](
            "the CUDA NVRTC library was not found at "
            + CUDA_CUBLASLT_LIBRARY_PATH
        )
    var ptr = Pointer[DLHandle].alloc(1)
    var handle = DLHandle(CUDA_CUBLASLT_LIBRARY_PATH)
    ptr[] = handle
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn _get_dylib_function[
    func_name: StringLiteral, result_type: AnyRegType
]() raises -> result_type:
    return _ffi_get_dylib_function[
        "CUDA_CUBLASLT_LIBRARY",
        func_name,
        _init_dylib,
        _destroy_dylib,
        result_type,
    ]()


# ===----------------------------------------------------------------------===#
# Bindings
# ===----------------------------------------------------------------------===#


fn cublasLtMatmulAlgoConfigSetAttribute(
    algo: Pointer[MatmulAlgorithm],
    attr: cublasLtMatmulAlgoConfigAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
) raises -> Result:
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
            Pointer[MatmulAlgorithm],
            cublasLtMatmulAlgoConfigAttributes_t,
            Pointer[NoneType],
            Int,
        ) raises -> Result,
    ]()(algo, attr, buf, size_in_bytes)


fn cublasLtCreate(
    light_handle: Pointer[Pointer[cublasLtContext]],
) raises -> Result:
    return _get_dylib_function[
        "cublasLtCreate",
        fn (Pointer[Pointer[cublasLtContext]]) raises -> Result,
    ]()(light_handle)


fn cublasLtMatrixTransformDescCreate(
    transform_desc: Pointer[Pointer[cublasLtMatrixTransformDescOpaque_t]],
    scale_type: DataType,
) raises -> Result:
    """Create new matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescCreate",
        fn (
            Pointer[Pointer[cublasLtMatrixTransformDescOpaque_t]],
            DataType,
        ) raises -> Result,
    ]()(transform_desc, scale_type)


@value
@register_passable("trivial")
struct Order:
    """Enum for data ordering ."""

    var _value: Int8
    alias COL = Order(0)
    """Column-major

    Leading dimension is the stride (in elements) to the beginning of next column in memory.
    """
    alias ROW = Order(1)
    """Row major

    Leading dimension is the stride (in elements) to the beginning of next row in memory.
    """
    alias COL32 = Order(2)
    """Column-major ordered tiles of 32 columns.

    Leading dimension is the stride (in elements) to the beginning of next group of 32-columns. E.g. if matrix has 33
    columns and 2 rows, ld must be at least (32) * 2 = 64.
    """
    alias COL4_4R2_8C = Order(3)
    """Column-major ordered tiles of composite tiles with total 32 columns and 8 rows, tile composed of interleaved
    inner tiles of 4 columns within 4 even or odd rows in an alternating pattern.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32 * 8) * 1 = 256.
    """
    alias COL32_2R_4R4 = Order(4)
    """Column-major ordered tiles of composite tiles with total 32 columns ands 32 rows.
    Element offset within the tile is calculated as (((row%8)/2*4+row/8)*2+row%2)*32+col.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 32 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32*32)*1 = 1024.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

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
        return int(self._value)


fn cublasLtMatrixLayoutSetAttribute(
    mat_layout: Pointer[cublasLtMatrixLayoutOpaque_t],
    attr: cublasLtMatrixLayoutAttribute_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
) raises -> Result:
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
            Pointer[cublasLtMatrixLayoutOpaque_t],
            cublasLtMatrixLayoutAttribute_t,
            Pointer[NoneType],
            Int,
        ) raises -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes)


@value
@register_passable("trivial")
struct ClusterShape:
    """Thread Block Cluster size.

    Typically dimensioned similar to cublasLtMatmulTile_t, with the third coordinate unused at this time.
    ."""

    var _value: Int8
    alias SHAPE_AUTO = ClusterShape(0)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x1x1 = ClusterShape(1)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x1x1 = ClusterShape(2)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x1x1 = ClusterShape(3)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x2x1 = ClusterShape(4)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x2x1 = ClusterShape(5)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x2x1 = ClusterShape(6)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x4x1 = ClusterShape(7)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x4x1 = ClusterShape(8)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x4x1 = ClusterShape(9)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_8x1x1 = ClusterShape(10)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x8x1 = ClusterShape(11)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_8x2x1 = ClusterShape(12)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x8x1 = ClusterShape(13)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_16x1x1 = ClusterShape(14)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x16x1 = ClusterShape(15)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x1x1 = ClusterShape(16)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x1x1 = ClusterShape(17)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_6x1x1 = ClusterShape(18)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_7x1x1 = ClusterShape(19)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_9x1x1 = ClusterShape(20)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_10x1x1 = ClusterShape(21)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_11x1x1 = ClusterShape(22)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_12x1x1 = ClusterShape(23)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_13x1x1 = ClusterShape(24)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_14x1x1 = ClusterShape(25)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_15x1x1 = ClusterShape(26)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x2x1 = ClusterShape(27)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x2x1 = ClusterShape(28)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_6x2x1 = ClusterShape(29)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_7x2x1 = ClusterShape(30)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x3x1 = ClusterShape(31)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x3x1 = ClusterShape(32)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x3x1 = ClusterShape(33)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_4x3x1 = ClusterShape(34)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_5x3x1 = ClusterShape(35)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x4x1 = ClusterShape(36)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x5x1 = ClusterShape(37)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x5x1 = ClusterShape(38)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_3x5x1 = ClusterShape(39)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x6x1 = ClusterShape(40)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x6x1 = ClusterShape(41)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x7x1 = ClusterShape(42)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_2x7x1 = ClusterShape(43)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x9x1 = ClusterShape(44)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x10x1 = ClusterShape(45)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x11x1 = ClusterShape(46)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x12x1 = ClusterShape(47)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x13x1 = ClusterShape(48)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x14x1 = ClusterShape(49)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_1x15x1 = ClusterShape(50)
    """Let library pick cluster shape automatically.
    """
    alias SHAPE_END = ClusterShape(51)
    """Let library pick cluster shape automatically.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

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
        return int(self._value)


fn cublasLtHeuristicsCacheSetCapacity(capacity: Int) raises -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheSetCapacity", fn (Int) raises -> Result
    ]()(capacity)


@value
@register_passable("trivial")
struct MatmulAlgorithmCapability:
    """Capabilities Attributes that can be retrieved from an initialized Algo structure
    ."""

    var _value: Int8
    alias SPLITK_SUPPORT = MatmulAlgorithmCapability(0)
    """support for split K, see CUBLASLT_ALGO_CONFIG_SPLITK_NUM

    int32_t, 0 means no support, supported otherwise.
    """
    alias REDUCTION_SCHEME_MASK = MatmulAlgorithmCapability(1)
    """reduction scheme mask, see cublasLtReductionScheme_t; shows supported reduction schemes, if reduction scheme is
    not masked out it is supported.

    e.g. int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) ==
    CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ? 1 : 0;

    uint32_t.
    """
    alias CTA_SWIZZLING_SUPPORT = MatmulAlgorithmCapability(2)
    """support for cta swizzling, see CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING

    uint32_t, 0 means no support, 1 means supported value of 1, other values are reserved.
    """
    alias STRIDED_BATCH_SUPPORT = MatmulAlgorithmCapability(3)
    """support strided batch

    int32_t, 0 means no support, supported otherwise.
    """
    alias OUT_OF_PLACE_RESULT_SUPPORT = MatmulAlgorithmCapability(4)
    """support results out of place (D != C in D = alpha.A.B + beta.C)

    int32_t, 0 means no support, supported otherwise.
    """
    alias UPLO_SUPPORT = MatmulAlgorithmCapability(5)
    """syrk/herk support (on top of regular gemm)

    int32_t, 0 means no support, supported otherwise.
    """
    alias TILE_IDS = MatmulAlgorithmCapability(6)
    """tile ids possible to use, see cublasLtMatmulTile_t; if no tile ids are supported use
    CUBLASLT_MATMUL_TILE_UNDEFINED

    use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count

    array of uint32_t.
    """
    alias CUSTOM_OPTION_MAX = MatmulAlgorithmCapability(7)
    """custom option range is from 0 to CUSTOM_OPTION_MAX (inclusive), see
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION

    int32_t.
    """
    alias CUSTOM_MEMORY_ORDER = MatmulAlgorithmCapability(8)
    """whether algorithm supports custom (not COL or ROW memory order), see Order

    int32_t 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different
    requirements;.
    """
    alias POINTER_MODE_MASK = MatmulAlgorithmCapability(9)
    """bitmask enumerating pointer modes algorithm supports

    uint32_t, see cublasLtPointerModeMask_t.
    """
    alias EPILOGUE_MASK = MatmulAlgorithmCapability(10)
    """bitmask enumerating kinds of postprocessing algorithm supports in the epilogue

    uint32_t, see cublasLtEpilogue_t.
    """
    alias STAGES_IDS = MatmulAlgorithmCapability(11)
    """stages ids possible to use, see cublasLtMatmulStages_t; if no stages ids are supported use
    CUBLASLT_MATMUL_STAGES_UNDEFINED

    use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count

    array of uint32_t.
    """
    alias LD_NEGATIVE = MatmulAlgorithmCapability(12)
    """support for nagative ld for all of the matrices

    int32_t 0 means no support, supported otherwise.
    """
    alias NUMERICAL_IMPL_FLAGS = MatmulAlgorithmCapability(13)
    """details about algorithm's implementation that affect it's numerical behavior

    uint64_t, see cublasLtNumericalImplFlags_t.
    """
    alias MIN_ALIGNMENT_A_BYTES = MatmulAlgorithmCapability(14)
    """minimum alignment required for A matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_B_BYTES = MatmulAlgorithmCapability(15)
    """minimum alignment required for B matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_C_BYTES = MatmulAlgorithmCapability(16)
    """minimum alignment required for C matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias MIN_ALIGNMENT_D_BYTES = MatmulAlgorithmCapability(17)
    """minimum alignment required for D matrix in bytes
    (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)

    uint32_t.
    """
    alias ATOMIC_SYNC = MatmulAlgorithmCapability(18)
    """EXPERIMENTAL: support for synchronization via atomic counters

    int32_t.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

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
        return int(self._value)


fn cublasLtGetStatusString(status: Result) raises -> Pointer[Int8]:
    return _get_dylib_function[
        "cublasLtGetStatusString", fn (Result) raises -> Pointer[Int8]
    ]()(status)


@value
@register_passable("trivial")
struct PointerMode:
    """Pointer mode to use for alpha/beta ."""

    var _value: Int8
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

    fn __init__(inout self, value: Int):
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
        return abort[String]("invalid PointerMode entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtMatmulDescGetAttribute(
    matmul_desc: Pointer[cublasLtMatmulDescOpaque_t],
    attr: cublasLtMatmulDescAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
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
            Pointer[cublasLtMatmulDescOpaque_t],
            cublasLtMatmulDescAttributes_t,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes, size_written)


# Opaque descriptor for matrix memory layout
# .
alias cublasLtMatrixLayout_t = Pointer[cublasLtMatrixLayoutOpaque_t]

# Opaque descriptor for cublasLtMatrixTransform() operation details
# .
alias cublasLtMatrixTransformDesc_t = Pointer[
    cublasLtMatrixTransformDescOpaque_t
]


fn cublasLtMatmulAlgoCheck(
    light_handle: Pointer[cublasLtContext],
    operation_desc: Pointer[cublasLtMatmulDescOpaque_t],
    _adesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _bdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _cdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _ddesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    algo: Pointer[MatmulAlgorithm],
    result: Pointer[cublasLtMatmulHeuristicResult_t],
) raises -> Result:
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
            Pointer[cublasLtContext],
            Pointer[cublasLtMatmulDescOpaque_t],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[MatmulAlgorithm],
            Pointer[cublasLtMatmulHeuristicResult_t],
        ) raises -> Result,
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


@value
@register_passable("trivial")
struct cublasLtMatmulSearch_t:
    """Matmul heuristic search mode
    ."""

    var _value: Int8
    alias CUBLASLT_SEARCH_BEST_FIT = cublasLtMatmulSearch_t(0)
    """ask heuristics for best algo for given usecase.
    """
    alias CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID = cublasLtMatmulSearch_t(1)
    """only try to find best config for preconfigured algo id.
    """
    alias CUBLASLT_SEARCH_RESERVED_02 = cublasLtMatmulSearch_t(2)
    """reserved for future use.
    """
    alias CUBLASLT_SEARCH_RESERVED_03 = cublasLtMatmulSearch_t(3)
    """reserved for future use.
    """
    alias CUBLASLT_SEARCH_RESERVED_04 = cublasLtMatmulSearch_t(4)
    """reserved for future use.
    """
    alias CUBLASLT_SEARCH_RESERVED_05 = cublasLtMatmulSearch_t(5)
    """reserved for future use.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_SEARCH_BEST_FIT:
            return "CUBLASLT_SEARCH_BEST_FIT"
        if self == Self.CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID:
            return "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID"
        if self == Self.CUBLASLT_SEARCH_RESERVED_02:
            return "CUBLASLT_SEARCH_RESERVED_02"
        if self == Self.CUBLASLT_SEARCH_RESERVED_03:
            return "CUBLASLT_SEARCH_RESERVED_03"
        if self == Self.CUBLASLT_SEARCH_RESERVED_04:
            return "CUBLASLT_SEARCH_RESERVED_04"
        if self == Self.CUBLASLT_SEARCH_RESERVED_05:
            return "CUBLASLT_SEARCH_RESERVED_05"
        return abort[String]("invalid cublasLtMatmulSearch_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct cublasLtReductionScheme_t:
    """Reduction scheme for portions of the dot-product calculated in parallel (a. k. a. "split - K").
    ."""

    var _value: Int8
    alias CUBLASLT_REDUCTION_SCHEME_NONE = cublasLtReductionScheme_t(0)
    """No reduction scheme, dot-product shall be performed in one sequence.
    """
    alias CUBLASLT_REDUCTION_SCHEME_INPLACE = cublasLtReductionScheme_t(1)
    """Reduction is performed "in place" - using the output buffer (and output data type) and counters (in workspace) to
    guarantee the sequentiality.
    """
    alias CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE = cublasLtReductionScheme_t(2)
    """Intermediate results are stored in compute type in the workspace and reduced in a separate step.
    """
    alias CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE = cublasLtReductionScheme_t(3)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """
    alias CUBLASLT_REDUCTION_SCHEME_MASK = cublasLtReductionScheme_t(4)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_REDUCTION_SCHEME_NONE:
            return "CUBLASLT_REDUCTION_SCHEME_NONE"
        if self == Self.CUBLASLT_REDUCTION_SCHEME_INPLACE:
            return "CUBLASLT_REDUCTION_SCHEME_INPLACE"
        if self == Self.CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE:
            return "CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE"
        if self == Self.CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE:
            return "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE"
        if self == Self.CUBLASLT_REDUCTION_SCHEME_MASK:
            return "CUBLASLT_REDUCTION_SCHEME_MASK"
        return abort[String]("invalid cublasLtReductionScheme_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtLoggerSetCallback(
    callback: fn (Int16, Pointer[Int8], Pointer[NoneType]) raises -> NoneType
) raises -> Result:
    """Experimental: Logger callback setter.

    callback                     a user defined callback function to be called by the logger

    \retval     CUBLAS_STATUS_SUCCESS        if callback was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetCallback",
        fn (
            fn (Int16, Pointer[Int8], Pointer[NoneType]) raises -> NoneType
        ) raises -> Result,
    ]()(callback)


fn cublasLtGetProperty(type: Property, value: Pointer[Int16]) raises -> Result:
    return _get_dylib_function[
        "cublasLtGetProperty",
        fn (Property, Pointer[Int16]) raises -> Result,
    ]()(type, value)


fn cublasLtGetVersion() raises -> Int:
    return _get_dylib_function["cublasLtGetVersion", fn () raises -> Int]()()


fn cublasLtMatrixLayoutGetAttribute(
    mat_layout: Pointer[cublasLtMatrixLayoutOpaque_t],
    attr: cublasLtMatrixLayoutAttribute_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
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
            Pointer[cublasLtMatrixLayoutOpaque_t],
            cublasLtMatrixLayoutAttribute_t,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes, size_written)


@register_passable("trivial")
struct cublasLtMatmulPreferenceOpaque_t:
    """Semi-opaque descriptor for cublasLtMatmulPreference() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@value
@register_passable("trivial")
struct cublasLtMatmulDescAttributes_t:
    """Matmul descriptor attributes to define details of the operation. ."""

    var _value: Int8
    alias CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = cublasLtMatmulDescAttributes_t(0)
    """Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
    accumulator during matrix multiplication.

    int32_t.
    """
    alias CUBLASLT_MATMUL_DESC_SCALE_TYPE = cublasLtMatmulDescAttributes_t(1)
    """Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
    typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
    D before being stored in memory.

    int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE.
    """
    alias CUBLASLT_MATMUL_DESC_POINTER_MODE = cublasLtMatmulDescAttributes_t(2)
    """Pointer mode of alpha and beta, see PointerMode. When DEVICE_VECTOR is in use,
    alpha/beta vector lenghts must match number of output matrix rows.

    int32_t, default: HOST.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSA = cublasLtMatmulDescAttributes_t(3)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSB = cublasLtMatmulDescAttributes_t(4)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_TRANSC = cublasLtMatmulDescAttributes_t(5)
    """Transform of matrix C, see cublasOperation_t.

    Currently only CUBLAS_OP_N is supported.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATMUL_DESC_FILL_MODE = cublasLtMatmulDescAttributes_t(6)
    """Matrix fill mode, see cublasFillMode_t.

    int32_t, default: CUBLAS_FILL_MODE_FULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE = cublasLtMatmulDescAttributes_t(7)
    """Epilogue function, see cublasLtEpilogue_t.

    uint32_t, default: CUBLASLT_EPILOGUE_DEFAULT.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_POINTER = cublasLtMatmulDescAttributes_t(8)
    """Bias or bias gradient vector pointer in the device memory.

    Bias case. See CUBLASLT_EPILOGUE_BIAS.
    For bias data type see CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE.

    Bias vector length must match matrix D rows count.

    Bias gradient case. See CUBLASLT_EPILOGUE_DRELU_BGRAD and CUBLASLT_EPILOGUE_DGELU_BGRAD.
    Bias gradient vector elements are the same type as the output elements
    (Ctype) with the exception of IMMA kernels (see above).

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Bias case: const void *, default: NULL
    Bias gradient case: void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = cublasLtMatmulDescAttributes_t(
        9
    )
    """Batch stride for bias or bias gradient vector.

    Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = cublasLtMatmulDescAttributes_t(
        10
    )
    """Pointer for epilogue auxiliary buffer.

    - Output vector for ReLu bit-mask in forward pass when CUBLASLT_EPILOGUE_RELU_AUX
     or CUBLASLT_EPILOGUE_RELU_AUX_BIAS epilogue is used.
    - Input vector for ReLu bit-mask in backward pass when
     CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used.

    - Output of GELU input matrix in forward pass when
     CUBLASLT_EPILOGUE_GELU_AUX_BIAS epilogue is used.
    - Input of GELU input matrix for backward pass when
     CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue is used.

    For aux data type see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE.

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.

    Forward pass: void *, default: NULL
    Backward pass: const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = cublasLtMatmulDescAttributes_t(
        11
    )
    """Leading dimension for epilogue auxiliary buffer.

    - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
     when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
    used. Must be divisible by 128 and be no less than the number of rows in the output matrix.

    - GELU input matrix leading dimension in elements
     when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
     Must be divisible by 8 and be no less than the number of rows in the output matrix.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = cublasLtMatmulDescAttributes_t(
        12
    )
    """Batch stride for epilogue auxiliary buffer.

    - ReLu bit-mask matrix batch stride in elements (i.e. bits)
     when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
    used. Must be divisible by 128.

    - GELU input matrix batch stride in elements
     when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
     Must be divisible by 8.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = cublasLtMatmulDescAttributes_t(
        13
    )
    """Batch stride for alpha vector.

    Used together with ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1. If ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesnt supported batched alpha vector.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = cublasLtMatmulDescAttributes_t(
        14
    )
    """Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs
    when user expects a concurrent stream to be using some of the device resources.

    int32_t, default: 0 - use the number reported by the device.
    """
    alias CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = cublasLtMatmulDescAttributes_t(
        15
    )
    """Device pointer to the scale factor value that converts data in matrix A to the compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = cublasLtMatmulDescAttributes_t(
        16
    )
    """Device pointer to the scale factor value to convert data in matrix B to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = cublasLtMatmulDescAttributes_t(
        17
    )
    """Device pointer to the scale factor value to convert data in matrix C to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = cublasLtMatmulDescAttributes_t(
        18
    )
    """Device pointer to the scale factor value to convert data in matrix D to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = cublasLtMatmulDescAttributes_t(
        19
    )
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    output matrix.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = cublasLtMatmulDescAttributes_t(
        20
    )
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
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = cublasLtMatmulDescAttributes_t(
        21
    )
    """Device pointer to the scaling factor value to convert results from compute type data range to storage
    data range in the auxiliary matrix that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data,
    scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = cublasLtMatmulDescAttributes_t(
        22
    )
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    buffer that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_FAST_ACCUM = cublasLtMatmulDescAttributes_t(23)
    """Flag for managing fp8 fast accumulation mode.
    When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results
    will not periodically be promoted to a higher precision.

    int8_t, default: 0 - fast accumulation mode is disabled.
    """
    alias CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = cublasLtMatmulDescAttributes_t(
        24
    )
    """Type of bias or bias gradient vector in the device memory.

    Bias case: see CUBLASLT_EPILOGUE_BIAS.

    Bias vector elements are the same type as the elements of output matrix (Dtype) with the following exceptions:
    - IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements
     are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
    - fp8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2, See
     https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul for details.

    int32_t based on cudaDataType, default: -1.
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = cublasLtMatmulDescAttributes_t(
        25
    )
    """EXPERIMENTAL: Number of atomic synchronization chunks in the row dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = cublasLtMatmulDescAttributes_t(
        26
    )
    """EXPERIMENTAL: Number of atomic synchronization chunks in the column dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = cublasLtMatmulDescAttributes_t(
        27
    )
    """EXPERIMENTAL: Pointer to a device array of input atomic counters consumed by a matmul.

    int32_t *, default: NULL.
    """
    alias CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = cublasLtMatmulDescAttributes_t(
        28
    )
    """EXPERIMENTAL: Pointer to a device array of output atomic counters produced by a matmul.

    int32_t *, default: NULL.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

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
        return int(self._value)


fn cublasLtMatrixTransformDescInit_internal(
    transform_desc: Pointer[cublasLtMatrixTransformDescOpaque_t],
    size: Int,
    scale_type: DataType,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescInit_internal",
        fn (
            Pointer[cublasLtMatrixTransformDescOpaque_t], Int, DataType
        ) raises -> Result,
    ]()(transform_desc, size, scale_type)


fn cublasLtMatrixLayoutDestroy(
    mat_layout: Pointer[cublasLtMatrixLayoutOpaque_t],
) raises -> Result:
    """Destroy matrix layout descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutDestroy",
        fn (Pointer[cublasLtMatrixLayoutOpaque_t]) raises -> Result,
    ]()(mat_layout)


# Opaque descriptor for cublasLtMatmul() operation details
# .
alias cublasLtMatmulDesc_t = Pointer[cublasLtMatmulDescOpaque_t]

# Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
# .
alias cublasLtMatmulPreference_t = Pointer[cublasLtMatmulPreferenceOpaque_t]


fn cublasLtMatmul(
    light_handle: Pointer[cublasLtContext],
    compute_desc: Pointer[cublasLtMatmulDescOpaque_t],
    alpha: Pointer[NoneType],
    _a: Pointer[NoneType],
    _adesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _b: Pointer[NoneType],
    _bdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    beta: Pointer[NoneType],
    _c: Pointer[NoneType],
    _cdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _d: Pointer[NoneType],
    _ddesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    algo: Pointer[MatmulAlgorithm],
    workspace: Pointer[NoneType],
    workspace_size_in_bytes: Int,
    stream: Pointer[Stream],
) raises -> Result:
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
            Pointer[cublasLtContext],
            Pointer[cublasLtMatmulDescOpaque_t],
            Pointer[NoneType],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[NoneType],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[MatmulAlgorithm],
            Pointer[NoneType],
            Int,
            Pointer[Stream],
        ) raises -> Result,
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
    transform_desc: Pointer[cublasLtMatrixTransformDescOpaque_t],
) raises -> Result:
    """Destroy matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescDestroy",
        fn (Pointer[cublasLtMatrixTransformDescOpaque_t]) raises -> Result,
    ]()(transform_desc)


fn cublasLtMatmulAlgoCapGetAttribute(
    algo: Pointer[MatmulAlgorithm],
    attr: MatmulAlgorithmCapability,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
    """Get algo capability attribute.

    E.g. to get list of supported Tile IDs:
        cublasLtMatmulTile_t tiles[CUBLASLT_MATMUL_TILE_END];
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
            Pointer[MatmulAlgorithm],
            MatmulAlgorithmCapability,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulDescSetAttribute(
    matmul_desc: Pointer[cublasLtMatmulDescOpaque_t],
    attr: cublasLtMatmulDescAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
) raises -> Result:
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
            Pointer[cublasLtMatmulDescOpaque_t],
            cublasLtMatmulDescAttributes_t,
            Pointer[NoneType],
            Int,
        ) raises -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes)


fn cublasLtMatmulPreferenceSetAttribute(
    pref: Pointer[cublasLtMatmulPreferenceOpaque_t],
    attr: cublasLtMatmulPreferenceAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
) raises -> Result:
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
            Pointer[cublasLtMatmulPreferenceOpaque_t],
            cublasLtMatmulPreferenceAttributes_t,
            Pointer[NoneType],
            Int,
        ) raises -> Result,
    ]()(pref, attr, buf, size_in_bytes)


# Experimental: Logger callback type.
# .
alias cublasLtLoggerCallback_t = fn (
    Int32, DTypePointer[DType.int8], DTypePointer[DType.int8]
) -> None


fn cublasLtMatrixLayoutInit_internal(
    mat_layout: Pointer[cublasLtMatrixLayoutOpaque_t],
    size: Int,
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutInit_internal",
        fn (
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Int,
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) raises -> Result,
    ]()(mat_layout, size, type, rows, cols, ld)


@value
@register_passable("trivial")
struct cublasLtMatmulPreferenceAttributes_t:
    """Algo search preference to fine tune the heuristic function. ."""

    var _value: Int8
    alias CUBLASLT_MATMUL_PREF_SEARCH_MODE = cublasLtMatmulPreferenceAttributes_t(
        0
    )
    """Search mode, see cublasLtMatmulSearch_t.

    uint32_t, default: CUBLASLT_SEARCH_BEST_FIT.
    """
    alias CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = cublasLtMatmulPreferenceAttributes_t(
        1
    )
    """Maximum allowed workspace size in bytes.

    uint64_t, default: 0 - no workspace allowed.
    """
    alias CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = cublasLtMatmulPreferenceAttributes_t(
        2
    )
    """Reduction scheme mask, see cublasLtReductionScheme_t. Filters heuristic result to only include algo configs that
    use one of the required modes.

    E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.

    uint32_t, default: CUBLASLT_REDUCTION_SCHEME_MASK (allows all reduction schemes).
    """
    alias CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = cublasLtMatmulPreferenceAttributes_t(
        3
    )
    """Minimum buffer alignment for matrix A (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = cublasLtMatmulPreferenceAttributes_t(
        4
    )
    """Minimum buffer alignment for matrix B (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = cublasLtMatmulPreferenceAttributes_t(
        5
    )
    """Minimum buffer alignment for matrix C (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = cublasLtMatmulPreferenceAttributes_t(
        6
    )
    """Minimum buffer alignment for matrix D (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    alias CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = cublasLtMatmulPreferenceAttributes_t(
        7
    )
    """Maximum wave count.

    See cublasLtMatmulHeuristicResult_t::wavesCount.

    Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.

    float, default: 0.0f.
    """
    alias CUBLASLT_MATMUL_PREF_IMPL_MASK = cublasLtMatmulPreferenceAttributes_t(
        8
    )
    """Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
    algorithms that use the allowed implementations.

    uint64_t, default: uint64_t(-1) (allow everything).
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATMUL_PREF_SEARCH_MODE:
            return "CUBLASLT_MATMUL_PREF_SEARCH_MODE"
        if self == Self.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            return "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES"
        if self == Self.CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK:
            return "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK"
        if self == Self.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES:
            return "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES"
        if self == Self.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES:
            return "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES"
        if self == Self.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES:
            return "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES"
        if self == Self.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES:
            return "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES"
        if self == Self.CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT:
            return "CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT"
        if self == Self.CUBLASLT_MATMUL_PREF_IMPL_MASK:
            return "CUBLASLT_MATMUL_PREF_IMPL_MASK"
        return abort[String](
            "invalid cublasLtMatmulPreferenceAttributes_t entry"
        )

    fn __int__(self) raises -> Int:
        return int(self._value)


@register_passable("trivial")
struct MatmulAlgorithm:
    """Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes).

    This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
    on selecting the right configuration again.
    """

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


alias cublasLtNumericalImplFlags_t = UInt64


@value
@register_passable("trivial")
struct cublasLtMatmulAlgoConfigAttributes_t:
    """Algo Configuration Attributes that can be set according to the Algo capabilities
    ."""

    var _value: Int8
    alias CUBLASLT_ALGO_CONFIG_ID = cublasLtMatmulAlgoConfigAttributes_t(0)
    """algorithm index, see cublasLtMatmulAlgoGetIds()

    readonly, set by cublasLtMatmulAlgoInit()
    int32_t.
    """
    alias CUBLASLT_ALGO_CONFIG_TILE_ID = cublasLtMatmulAlgoConfigAttributes_t(1)
    """tile id, see cublasLtMatmulTile_t

    uint32_t, default: CUBLASLT_MATMUL_TILE_UNDEFINED.
    """
    alias CUBLASLT_ALGO_CONFIG_SPLITK_NUM = cublasLtMatmulAlgoConfigAttributes_t(
        2
    )
    """Number of K splits. If the number of K splits is greater than one, SPLITK_NUM parts
    of matrix multiplication will be computed in parallel. The results will be accumulated
    according to CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME

    int32_t, default: 1.
    """
    alias CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = cublasLtMatmulAlgoConfigAttributes_t(
        3
    )
    """reduction scheme, see cublasLtReductionScheme_t

    uint32_t, default: CUBLASLT_REDUCTION_SCHEME_NONE.
    """
    alias CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING = cublasLtMatmulAlgoConfigAttributes_t(
        4
    )
    """cta swizzling, change mapping from CUDA grid coordinates to parts of the matrices

    possible values: 0, 1, other values reserved

    uint32_t, default: 0.
    """
    alias CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION = cublasLtMatmulAlgoConfigAttributes_t(
        5
    )
    """custom option, each algorithm can support some custom options that don't fit description of the other config
    attributes, see CUSTOM_OPTION_MAX to get accepted range for any specific case

    uint32_t, default: 0.
    """
    alias CUBLASLT_ALGO_CONFIG_STAGES_ID = cublasLtMatmulAlgoConfigAttributes_t(
        6
    )
    """stages id, see cublasLtMatmulStages_t

    uint32_t, default: CUBLASLT_MATMUL_STAGES_UNDEFINED.
    """
    alias CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID = cublasLtMatmulAlgoConfigAttributes_t(
        7
    )
    """inner shape id, see InnerShape

    uint16_t, default: 0 (UNDEFINED).
    """
    alias CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID = cublasLtMatmulAlgoConfigAttributes_t(
        8
    )
    """Thread Block Cluster shape id, see ClusterShape. Defines cluster size to use.

    uint16_t, default: 0 (SHAPE_AUTO).
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_ALGO_CONFIG_ID:
            return "CUBLASLT_ALGO_CONFIG_ID"
        if self == Self.CUBLASLT_ALGO_CONFIG_TILE_ID:
            return "CUBLASLT_ALGO_CONFIG_TILE_ID"
        if self == Self.CUBLASLT_ALGO_CONFIG_SPLITK_NUM:
            return "CUBLASLT_ALGO_CONFIG_SPLITK_NUM"
        if self == Self.CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME:
            return "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME"
        if self == Self.CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING:
            return "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING"
        if self == Self.CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION:
            return "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION"
        if self == Self.CUBLASLT_ALGO_CONFIG_STAGES_ID:
            return "CUBLASLT_ALGO_CONFIG_STAGES_ID"
        if self == Self.CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID:
            return "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID"
        if self == Self.CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID:
            return "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID"
        return abort[String](
            "invalid cublasLtMatmulAlgoConfigAttributes_t entry"
        )

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtMatmulPreferenceDestroy(
    pref: Pointer[cublasLtMatmulPreferenceOpaque_t],
) raises -> Result:
    """Destroy matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceDestroy",
        fn (Pointer[cublasLtMatmulPreferenceOpaque_t]) raises -> Result,
    ]()(pref)


# fn cublasLtMatmulAlgoGetHeuristic(
#     light_handle: Pointer[cublasLtContext],
#     operation_desc: Pointer[cublasLtMatmulDescOpaque_t],
#     _adesc: Pointer[cublasLtMatrixLayoutOpaque_t],
#     _bdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
#     _cdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
#     _ddesc: Pointer[cublasLtMatrixLayoutOpaque_t],
#     preference: Pointer[cublasLtMatmulPreferenceOpaque_t],
#     requested_algo_count: Int16,
#     heuristic_results_array: UNKNOWN,
#     return_algo_count: Pointer[Int16],
# ) raises -> Result:
#     """Query cublasLt heuristic for algorithm appropriate for given use case.

#         lightHandle            Pointer to the allocated cuBLASLt handle for the cuBLASLt
#                                           context. See cublasLtHandle_t.
#         operationDesc          Handle to the matrix multiplication descriptor.
#         Adesc                  Handle to the layout descriptors for matrix A.
#         Bdesc                  Handle to the layout descriptors for matrix B.
#         Cdesc                  Handle to the layout descriptors for matrix C.
#         Ddesc                  Handle to the layout descriptors for matrix D.
#         preference             Pointer to the structure holding the heuristic search
#                                           preferences descriptor. See cublasLtMatrixLayout_t.
#         requestedAlgoCount     Size of heuristicResultsArray (in elements) and requested
#                                           maximum number of algorithms to return.
#     \param[in, out] heuristicResultsArray  Output algorithms and associated runtime characteristics,
#                                           ordered in increasing estimated compute time.
#         returnAlgoCount        The number of heuristicResultsArray elements written.

#     \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal to zero
#     \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function available for current configuration
#     \retval  CUBLAS_STATUS_SUCCESS         if query was successful, inspect
#                                           heuristicResultsArray[0 to (returnAlgoCount - 1)].state
#                                           for detail status of results
#     ."""
#     return _get_dylib_function[
#         "cublasLtMatmulAlgoGetHeuristic",
#         fn (
#             Pointer[cublasLtContext],
#             Pointer[cublasLtMatmulDescOpaque_t],
#             Pointer[cublasLtMatrixLayoutOpaque_t],
#             Pointer[cublasLtMatrixLayoutOpaque_t],
#             Pointer[cublasLtMatrixLayoutOpaque_t],
#             Pointer[cublasLtMatrixLayoutOpaque_t],
#             Pointer[cublasLtMatmulPreferenceOpaque_t],
#             Int16,
#             UNKNOWN,
#             Pointer[Int16],
#         ) raises -> Result,
#     ]()(
#         light_handle,
#         operation_desc,
#         _adesc,
#         _bdesc,
#         _cdesc,
#         _ddesc,
#         preference,
#         requested_algo_count,
#         heuristic_results_array,
#         return_algo_count,
#     )


@value
@register_passable("trivial")
struct InnerShape:
    """Inner size of the kernel.

    Represents various aspects of internal kernel design, that don't impact CUDA grid size but may have other more subtle
    effects.
    """

    var _value: Int8
    alias UNDEFINED = InnerShape(0)
    alias MMA884 = InnerShape(1)
    alias MMA1684 = InnerShape(2)
    alias MMA1688 = InnerShape(3)
    alias MMA16816 = InnerShape(4)
    alias END = InnerShape(5)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

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
        return int(self._value)


@value
@register_passable("trivial")
struct cublasLtMatrixLayoutAttribute_t:
    """Attributes of memory layout ."""

    var _value: Int8
    alias CUBLASLT_MATRIX_LAYOUT_TYPE = cublasLtMatrixLayoutAttribute_t(0)
    """Data type, see cudaDataType.

    uint32_t.
    """
    alias CUBLASLT_MATRIX_LAYOUT_ORDER = cublasLtMatrixLayoutAttribute_t(1)
    """Memory order of the data, see Order.

    int32_t, default: COL.
    """
    alias CUBLASLT_MATRIX_LAYOUT_ROWS = cublasLtMatrixLayoutAttribute_t(2)
    """Number of rows.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    alias CUBLASLT_MATRIX_LAYOUT_COLS = cublasLtMatrixLayoutAttribute_t(3)
    """Number of columns.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    alias CUBLASLT_MATRIX_LAYOUT_LD = cublasLtMatrixLayoutAttribute_t(4)
    """Matrix leading dimension.

    For COL this is stride (in elements) of matrix column, for more details and documentation for
    other memory orders see documentation for Order values.

    Currently only non-negative values are supported, must be large enough so that matrix memory locations are not
    overlapping (e.g. greater or equal to CUBLASLT_MATRIX_LAYOUT_ROWS in case of COL).

    int64_t;.
    """
    alias CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = cublasLtMatrixLayoutAttribute_t(
        5
    )
    """Number of matmul operations to perform in the batch.

    See also STRIDED_BATCH_SUPPORT

    int32_t, default: 1.
    """
    alias CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = cublasLtMatrixLayoutAttribute_t(
        6
    )
    """Stride (in elements) to the next matrix for strided batch operation.

    When matrix type is planar-complex (CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET != 0), batch stride
    is interpreted by cublasLtMatmul() in number of real valued sub-elements. E.g. for data of type CUDA_C_16F,
    offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices
    is a 2B (16bit) floating point type).

    NOTE: A bug in cublasLtMatrixTransform() causes it to interpret the batch stride for a planar-complex matrix
    as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride
    value 256 when calling cublasLtMatrixTransform() (each complex element is 4B with real and imaginary values 2B
    each). This behavior is expected to be corrected in the next major cuBLAS version.

    int64_t, default: 0.
    """
    alias CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = cublasLtMatrixLayoutAttribute_t(
        7
    )
    """Stride (in bytes) to the imaginary plane for planar complex layout.

    int64_t, default: 0 - 0 means that layout is regular (real and imaginary parts of complex numbers are interleaved
    in memory in each element).
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATRIX_LAYOUT_TYPE:
            return "CUBLASLT_MATRIX_LAYOUT_TYPE"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_ORDER:
            return "CUBLASLT_MATRIX_LAYOUT_ORDER"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_ROWS:
            return "CUBLASLT_MATRIX_LAYOUT_ROWS"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_COLS:
            return "CUBLASLT_MATRIX_LAYOUT_COLS"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_LD:
            return "CUBLASLT_MATRIX_LAYOUT_LD"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            return "CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            return "CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET"
        if self == Self.CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET:
            return "CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET"
        return abort[String]("invalid cublasLtMatrixLayoutAttribute_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtDestroy(light_handle: Pointer[cublasLtContext]) raises -> Result:
    return _get_dylib_function[
        "cublasLtDestroy", fn (Pointer[cublasLtContext]) raises -> Result
    ]()(light_handle)


fn cublasLtGetCudartVersion() raises -> Int:
    return _get_dylib_function[
        "cublasLtGetCudartVersion", fn () raises -> Int
    ]()()


fn cublasLtMatmulAlgoConfigGetAttribute(
    algo: Pointer[MatmulAlgorithm],
    attr: cublasLtMatmulAlgoConfigAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
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
            Pointer[MatmulAlgorithm],
            cublasLtMatmulAlgoConfigAttributes_t,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


fn cublasLtLoggerForceDisable() raises -> Result:
    """Experimental: Disable logging for the entire session.

    \retval     CUBLAS_STATUS_SUCCESS        if disabled logging
    ."""
    return _get_dylib_function[
        "cublasLtLoggerForceDisable", fn () raises -> Result
    ]()()


fn cublasLtHeuristicsCacheGetCapacity(capacity: Pointer[Int]) raises -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheGetCapacity",
        fn (Pointer[Int]) raises -> Result,
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


fn cublasLtLoggerSetLevel(level: Int16) raises -> Result:
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
        "cublasLtLoggerSetLevel", fn (Int16) raises -> Result
    ]()(level)


@value
@register_passable("trivial")
struct cublasLtMatmulStages_t:
    """Size and number of stages in which elements are read into shared memory.

    General order of stages IDs is sorted by stage size first and by number of stages second.
    ."""

    var _value: Int8
    alias CUBLASLT_MATMUL_STAGES_UNDEFINED = cublasLtMatmulStages_t(0)
    alias CUBLASLT_MATMUL_STAGES_16x1 = cublasLtMatmulStages_t(1)
    alias CUBLASLT_MATMUL_STAGES_16x2 = cublasLtMatmulStages_t(2)
    alias CUBLASLT_MATMUL_STAGES_16x3 = cublasLtMatmulStages_t(3)
    alias CUBLASLT_MATMUL_STAGES_16x4 = cublasLtMatmulStages_t(4)
    alias CUBLASLT_MATMUL_STAGES_16x5 = cublasLtMatmulStages_t(5)
    alias CUBLASLT_MATMUL_STAGES_16x6 = cublasLtMatmulStages_t(6)
    alias CUBLASLT_MATMUL_STAGES_32x1 = cublasLtMatmulStages_t(7)
    alias CUBLASLT_MATMUL_STAGES_32x2 = cublasLtMatmulStages_t(8)
    alias CUBLASLT_MATMUL_STAGES_32x3 = cublasLtMatmulStages_t(9)
    alias CUBLASLT_MATMUL_STAGES_32x4 = cublasLtMatmulStages_t(10)
    alias CUBLASLT_MATMUL_STAGES_32x5 = cublasLtMatmulStages_t(11)
    alias CUBLASLT_MATMUL_STAGES_32x6 = cublasLtMatmulStages_t(12)
    alias CUBLASLT_MATMUL_STAGES_64x1 = cublasLtMatmulStages_t(13)
    alias CUBLASLT_MATMUL_STAGES_64x2 = cublasLtMatmulStages_t(14)
    alias CUBLASLT_MATMUL_STAGES_64x3 = cublasLtMatmulStages_t(15)
    alias CUBLASLT_MATMUL_STAGES_64x4 = cublasLtMatmulStages_t(16)
    alias CUBLASLT_MATMUL_STAGES_64x5 = cublasLtMatmulStages_t(17)
    alias CUBLASLT_MATMUL_STAGES_64x6 = cublasLtMatmulStages_t(18)
    alias CUBLASLT_MATMUL_STAGES_128x1 = cublasLtMatmulStages_t(19)
    alias CUBLASLT_MATMUL_STAGES_128x2 = cublasLtMatmulStages_t(20)
    alias CUBLASLT_MATMUL_STAGES_128x3 = cublasLtMatmulStages_t(21)
    alias CUBLASLT_MATMUL_STAGES_128x4 = cublasLtMatmulStages_t(22)
    alias CUBLASLT_MATMUL_STAGES_128x5 = cublasLtMatmulStages_t(23)
    alias CUBLASLT_MATMUL_STAGES_128x6 = cublasLtMatmulStages_t(24)
    alias CUBLASLT_MATMUL_STAGES_32x10 = cublasLtMatmulStages_t(25)
    alias CUBLASLT_MATMUL_STAGES_8x4 = cublasLtMatmulStages_t(26)
    alias CUBLASLT_MATMUL_STAGES_16x10 = cublasLtMatmulStages_t(27)
    alias CUBLASLT_MATMUL_STAGES_8x5 = cublasLtMatmulStages_t(28)
    alias CUBLASLT_MATMUL_STAGES_8x3 = cublasLtMatmulStages_t(29)
    alias CUBLASLT_MATMUL_STAGES_8xAUTO = cublasLtMatmulStages_t(30)
    alias CUBLASLT_MATMUL_STAGES_16xAUTO = cublasLtMatmulStages_t(31)
    alias CUBLASLT_MATMUL_STAGES_32xAUTO = cublasLtMatmulStages_t(32)
    alias CUBLASLT_MATMUL_STAGES_64xAUTO = cublasLtMatmulStages_t(33)
    alias CUBLASLT_MATMUL_STAGES_128xAUTO = cublasLtMatmulStages_t(34)
    alias CUBLASLT_MATMUL_STAGES_END = cublasLtMatmulStages_t(35)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATMUL_STAGES_UNDEFINED:
            return "CUBLASLT_MATMUL_STAGES_UNDEFINED"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x1:
            return "CUBLASLT_MATMUL_STAGES_16x1"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x2:
            return "CUBLASLT_MATMUL_STAGES_16x2"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x3:
            return "CUBLASLT_MATMUL_STAGES_16x3"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x4:
            return "CUBLASLT_MATMUL_STAGES_16x4"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x5:
            return "CUBLASLT_MATMUL_STAGES_16x5"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x6:
            return "CUBLASLT_MATMUL_STAGES_16x6"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x1:
            return "CUBLASLT_MATMUL_STAGES_32x1"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x2:
            return "CUBLASLT_MATMUL_STAGES_32x2"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x3:
            return "CUBLASLT_MATMUL_STAGES_32x3"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x4:
            return "CUBLASLT_MATMUL_STAGES_32x4"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x5:
            return "CUBLASLT_MATMUL_STAGES_32x5"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x6:
            return "CUBLASLT_MATMUL_STAGES_32x6"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x1:
            return "CUBLASLT_MATMUL_STAGES_64x1"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x2:
            return "CUBLASLT_MATMUL_STAGES_64x2"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x3:
            return "CUBLASLT_MATMUL_STAGES_64x3"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x4:
            return "CUBLASLT_MATMUL_STAGES_64x4"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x5:
            return "CUBLASLT_MATMUL_STAGES_64x5"
        if self == Self.CUBLASLT_MATMUL_STAGES_64x6:
            return "CUBLASLT_MATMUL_STAGES_64x6"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x1:
            return "CUBLASLT_MATMUL_STAGES_128x1"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x2:
            return "CUBLASLT_MATMUL_STAGES_128x2"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x3:
            return "CUBLASLT_MATMUL_STAGES_128x3"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x4:
            return "CUBLASLT_MATMUL_STAGES_128x4"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x5:
            return "CUBLASLT_MATMUL_STAGES_128x5"
        if self == Self.CUBLASLT_MATMUL_STAGES_128x6:
            return "CUBLASLT_MATMUL_STAGES_128x6"
        if self == Self.CUBLASLT_MATMUL_STAGES_32x10:
            return "CUBLASLT_MATMUL_STAGES_32x10"
        if self == Self.CUBLASLT_MATMUL_STAGES_8x4:
            return "CUBLASLT_MATMUL_STAGES_8x4"
        if self == Self.CUBLASLT_MATMUL_STAGES_16x10:
            return "CUBLASLT_MATMUL_STAGES_16x10"
        if self == Self.CUBLASLT_MATMUL_STAGES_8x5:
            return "CUBLASLT_MATMUL_STAGES_8x5"
        if self == Self.CUBLASLT_MATMUL_STAGES_8x3:
            return "CUBLASLT_MATMUL_STAGES_8x3"
        if self == Self.CUBLASLT_MATMUL_STAGES_8xAUTO:
            return "CUBLASLT_MATMUL_STAGES_8xAUTO"
        if self == Self.CUBLASLT_MATMUL_STAGES_16xAUTO:
            return "CUBLASLT_MATMUL_STAGES_16xAUTO"
        if self == Self.CUBLASLT_MATMUL_STAGES_32xAUTO:
            return "CUBLASLT_MATMUL_STAGES_32xAUTO"
        if self == Self.CUBLASLT_MATMUL_STAGES_64xAUTO:
            return "CUBLASLT_MATMUL_STAGES_64xAUTO"
        if self == Self.CUBLASLT_MATMUL_STAGES_128xAUTO:
            return "CUBLASLT_MATMUL_STAGES_128xAUTO"
        if self == Self.CUBLASLT_MATMUL_STAGES_END:
            return "CUBLASLT_MATMUL_STAGES_END"
        return abort[String]("invalid cublasLtMatmulStages_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtMatmulDescDestroy(
    matmul_desc: Pointer[cublasLtMatmulDescOpaque_t],
) raises -> Result:
    """Destroy matmul operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescDestroy",
        fn (Pointer[cublasLtMatmulDescOpaque_t]) raises -> Result,
    ]()(matmul_desc)


fn cublasLtMatrixTransformDescSetAttribute(
    transform_desc: Pointer[cublasLtMatrixTransformDescOpaque_t],
    attr: cublasLtMatrixTransformDescAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
) raises -> Result:
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
            Pointer[cublasLtMatrixTransformDescOpaque_t],
            cublasLtMatrixTransformDescAttributes_t,
            Pointer[NoneType],
            Int,
        ) raises -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes)


fn cublasLtMatmulPreferenceGetAttribute(
    pref: Pointer[cublasLtMatmulPreferenceOpaque_t],
    attr: cublasLtMatmulPreferenceAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
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
            Pointer[cublasLtMatmulPreferenceOpaque_t],
            cublasLtMatmulPreferenceAttributes_t,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(pref, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulAlgoInit(
    light_handle: Pointer[cublasLtContext],
    compute_type: ComputeType,
    scale_type: DataType,
    _atype: DataType,
    _btype: DataType,
    _ctype: DataType,
    _dtype: DataType,
    algo_id: Int16,
    algo: Pointer[MatmulAlgorithm],
) raises -> Result:
    """Initialize algo structure.

    \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside of recognized range
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not supported for given combination of data types
    \retval     CUBLAS_STATUS_SUCCESS        if the structure was successfully initialized
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoInit",
        fn (
            Pointer[cublasLtContext],
            ComputeType,
            DataType,
            DataType,
            DataType,
            DataType,
            DataType,
            Int16,
            Pointer[MatmulAlgorithm],
        ) raises -> Result,
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


@value
@register_passable("trivial")
struct cublasLtEpilogue_t:
    """Postprocessing options for the epilogue
    ."""

    var _value: Int8
    alias CUBLASLT_EPILOGUE_DEFAULT = cublasLtEpilogue_t(0)
    """No special postprocessing, just scale and quantize results if necessary.
    """
    alias CUBLASLT_EPILOGUE_RELU = cublasLtEpilogue_t(1)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
    """
    alias CUBLASLT_EPILOGUE_RELU_AUX = cublasLtEpilogue_t(2)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_BIAS = cublasLtEpilogue_t(3)
    """Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed
    (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final
    postprocessing.
    """
    alias CUBLASLT_EPILOGUE_RELU_BIAS = cublasLtEpilogue_t(4)
    """ReLu and Bias, apply Bias and then ReLu transform.
    """
    alias CUBLASLT_EPILOGUE_RELU_AUX_BIAS = cublasLtEpilogue_t(5)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_DRELU = cublasLtEpilogue_t(6)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_DRELU_BGRAD = cublasLtEpilogue_t(7)
    """ReLu and Bias, apply Bias and then ReLu transform

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_GELU = cublasLtEpilogue_t(8)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    """
    alias CUBLASLT_EPILOGUE_GELU_AUX = cublasLtEpilogue_t(9)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_GELU_BIAS = cublasLtEpilogue_t(10)
    """GELU and Bias, apply Bias and then GELU transform.
    """
    alias CUBLASLT_EPILOGUE_GELU_AUX_BIAS = cublasLtEpilogue_t(11)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_DGELU = cublasLtEpilogue_t(12)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_DGELU_BGRAD = cublasLtEpilogue_t(13)
    """GELU and Bias, apply Bias and then GELU transform

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    alias CUBLASLT_EPILOGUE_BGRADA = cublasLtEpilogue_t(14)
    """Bias gradient based on the input matrix A.

    The bias size corresponds to the number of rows of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """
    alias CUBLASLT_EPILOGUE_BGRADB = cublasLtEpilogue_t(15)
    """Bias gradient based on the input matrix B.

    The bias size corresponds to the number of columns of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_EPILOGUE_DEFAULT:
            return "CUBLASLT_EPILOGUE_DEFAULT"
        if self == Self.CUBLASLT_EPILOGUE_RELU:
            return "CUBLASLT_EPILOGUE_RELU"
        if self == Self.CUBLASLT_EPILOGUE_RELU_AUX:
            return "CUBLASLT_EPILOGUE_RELU_AUX"
        if self == Self.CUBLASLT_EPILOGUE_BIAS:
            return "CUBLASLT_EPILOGUE_BIAS"
        if self == Self.CUBLASLT_EPILOGUE_RELU_BIAS:
            return "CUBLASLT_EPILOGUE_RELU_BIAS"
        if self == Self.CUBLASLT_EPILOGUE_RELU_AUX_BIAS:
            return "CUBLASLT_EPILOGUE_RELU_AUX_BIAS"
        if self == Self.CUBLASLT_EPILOGUE_DRELU:
            return "CUBLASLT_EPILOGUE_DRELU"
        if self == Self.CUBLASLT_EPILOGUE_DRELU_BGRAD:
            return "CUBLASLT_EPILOGUE_DRELU_BGRAD"
        if self == Self.CUBLASLT_EPILOGUE_GELU:
            return "CUBLASLT_EPILOGUE_GELU"
        if self == Self.CUBLASLT_EPILOGUE_GELU_AUX:
            return "CUBLASLT_EPILOGUE_GELU_AUX"
        if self == Self.CUBLASLT_EPILOGUE_GELU_BIAS:
            return "CUBLASLT_EPILOGUE_GELU_BIAS"
        if self == Self.CUBLASLT_EPILOGUE_GELU_AUX_BIAS:
            return "CUBLASLT_EPILOGUE_GELU_AUX_BIAS"
        if self == Self.CUBLASLT_EPILOGUE_DGELU:
            return "CUBLASLT_EPILOGUE_DGELU"
        if self == Self.CUBLASLT_EPILOGUE_DGELU_BGRAD:
            return "CUBLASLT_EPILOGUE_DGELU_BGRAD"
        if self == Self.CUBLASLT_EPILOGUE_BGRADA:
            return "CUBLASLT_EPILOGUE_BGRADA"
        if self == Self.CUBLASLT_EPILOGUE_BGRADB:
            return "CUBLASLT_EPILOGUE_BGRADB"
        return abort[String]("invalid cublasLtEpilogue_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


@register_passable("trivial")
struct cublasLtMatmulDescOpaque_t:
    """Semi-opaque descriptor for cublasLtMatmul() operation details
    ."""

    var data: StaticTuple[UInt64, 32]


fn cublasLtMatrixLayoutCreate(
    mat_layout: Pointer[Pointer[cublasLtMatrixLayoutOpaque_t]],
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Result:
    """Create new matrix layout descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutCreate",
        fn (
            Pointer[Pointer[cublasLtMatrixLayoutOpaque_t]],
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) raises -> Result,
    ]()(mat_layout, type, rows, cols, ld)


@value
@register_passable("trivial")
struct cublasLtPointerModeMask_t:
    """Mask to define pointer mode capability ."""

    var _value: Int8
    alias MASK_HOST = cublasLtPointerModeMask_t(0)
    """see HOST.
    """
    alias MASK_DEVICE = cublasLtPointerModeMask_t(1)
    """see DEVICE.
    """
    alias MASK_DEVICE_VECTOR = cublasLtPointerModeMask_t(2)
    """see DEVICE_VECTOR.
    """
    alias MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO = cublasLtPointerModeMask_t(3)
    """see ALPHA_DEVICE_VECTOR_BETA_ZERO.
    """
    alias MASK_ALPHA_DEVICE_VECTOR_BETA_HOST = cublasLtPointerModeMask_t(4)
    """see ALPHA_DEVICE_VECTOR_BETA_HOST.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.MASK_HOST:
            return "MASK_HOST"
        if self == Self.MASK_DEVICE:
            return "MASK_DEVICE"
        if self == Self.MASK_DEVICE_VECTOR:
            return "MASK_DEVICE_VECTOR"
        if self == Self.MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO:
            return "MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO"
        if self == Self.MASK_ALPHA_DEVICE_VECTOR_BETA_HOST:
            return "MASK_ALPHA_DEVICE_VECTOR_BETA_HOST"
        return abort[String]("invalid cublasLtPointerModeMask_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


@register_passable("trivial")
struct cublasLtMatrixLayoutOpaque_t:
    """Semi-opaque descriptor for matrix memory layout
    ."""

    var data: StaticTuple[UInt64, 8]


fn cublasLtMatmulDescCreate(
    matmul_desc: Pointer[Pointer[cublasLtMatmulDescOpaque_t]],
    compute_type: ComputeType,
    scale_type: DataType,
) raises -> Result:
    """Create new matmul operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescCreate",
        fn (
            Pointer[Pointer[cublasLtMatmulDescOpaque_t]],
            ComputeType,
            DataType,
        ) raises -> Result,
    ]()(matmul_desc, compute_type, scale_type)


@value
@register_passable("trivial")
struct cublasLtMatmulTile_t:
    """Tile size (in C/D matrix Rows x Cols).

    General order of tile IDs is sorted by size first and by first dimension second.
    ."""

    var _value: Int8
    alias CUBLASLT_MATMUL_TILE_UNDEFINED = cublasLtMatmulTile_t(0)
    alias CUBLASLT_MATMUL_TILE_8x8 = cublasLtMatmulTile_t(1)
    alias CUBLASLT_MATMUL_TILE_8x16 = cublasLtMatmulTile_t(2)
    alias CUBLASLT_MATMUL_TILE_16x8 = cublasLtMatmulTile_t(3)
    alias CUBLASLT_MATMUL_TILE_8x32 = cublasLtMatmulTile_t(4)
    alias CUBLASLT_MATMUL_TILE_16x16 = cublasLtMatmulTile_t(5)
    alias CUBLASLT_MATMUL_TILE_32x8 = cublasLtMatmulTile_t(6)
    alias CUBLASLT_MATMUL_TILE_8x64 = cublasLtMatmulTile_t(7)
    alias CUBLASLT_MATMUL_TILE_16x32 = cublasLtMatmulTile_t(8)
    alias CUBLASLT_MATMUL_TILE_32x16 = cublasLtMatmulTile_t(9)
    alias CUBLASLT_MATMUL_TILE_64x8 = cublasLtMatmulTile_t(10)
    alias CUBLASLT_MATMUL_TILE_32x32 = cublasLtMatmulTile_t(11)
    alias CUBLASLT_MATMUL_TILE_32x64 = cublasLtMatmulTile_t(12)
    alias CUBLASLT_MATMUL_TILE_64x32 = cublasLtMatmulTile_t(13)
    alias CUBLASLT_MATMUL_TILE_32x128 = cublasLtMatmulTile_t(14)
    alias CUBLASLT_MATMUL_TILE_64x64 = cublasLtMatmulTile_t(15)
    alias CUBLASLT_MATMUL_TILE_128x32 = cublasLtMatmulTile_t(16)
    alias CUBLASLT_MATMUL_TILE_64x128 = cublasLtMatmulTile_t(17)
    alias CUBLASLT_MATMUL_TILE_128x64 = cublasLtMatmulTile_t(18)
    alias CUBLASLT_MATMUL_TILE_64x256 = cublasLtMatmulTile_t(19)
    alias CUBLASLT_MATMUL_TILE_128x128 = cublasLtMatmulTile_t(20)
    alias CUBLASLT_MATMUL_TILE_256x64 = cublasLtMatmulTile_t(21)
    alias CUBLASLT_MATMUL_TILE_64x512 = cublasLtMatmulTile_t(22)
    alias CUBLASLT_MATMUL_TILE_128x256 = cublasLtMatmulTile_t(23)
    alias CUBLASLT_MATMUL_TILE_256x128 = cublasLtMatmulTile_t(24)
    alias CUBLASLT_MATMUL_TILE_512x64 = cublasLtMatmulTile_t(25)
    alias CUBLASLT_MATMUL_TILE_64x96 = cublasLtMatmulTile_t(26)
    alias CUBLASLT_MATMUL_TILE_96x64 = cublasLtMatmulTile_t(27)
    alias CUBLASLT_MATMUL_TILE_96x128 = cublasLtMatmulTile_t(28)
    alias CUBLASLT_MATMUL_TILE_128x160 = cublasLtMatmulTile_t(29)
    alias CUBLASLT_MATMUL_TILE_160x128 = cublasLtMatmulTile_t(30)
    alias CUBLASLT_MATMUL_TILE_192x128 = cublasLtMatmulTile_t(31)
    alias CUBLASLT_MATMUL_TILE_128x192 = cublasLtMatmulTile_t(32)
    alias CUBLASLT_MATMUL_TILE_128x96 = cublasLtMatmulTile_t(33)
    alias CUBLASLT_MATMUL_TILE_32x256 = cublasLtMatmulTile_t(34)
    alias CUBLASLT_MATMUL_TILE_256x32 = cublasLtMatmulTile_t(35)
    alias CUBLASLT_MATMUL_TILE_END = cublasLtMatmulTile_t(36)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATMUL_TILE_UNDEFINED:
            return "CUBLASLT_MATMUL_TILE_UNDEFINED"
        if self == Self.CUBLASLT_MATMUL_TILE_8x8:
            return "CUBLASLT_MATMUL_TILE_8x8"
        if self == Self.CUBLASLT_MATMUL_TILE_8x16:
            return "CUBLASLT_MATMUL_TILE_8x16"
        if self == Self.CUBLASLT_MATMUL_TILE_16x8:
            return "CUBLASLT_MATMUL_TILE_16x8"
        if self == Self.CUBLASLT_MATMUL_TILE_8x32:
            return "CUBLASLT_MATMUL_TILE_8x32"
        if self == Self.CUBLASLT_MATMUL_TILE_16x16:
            return "CUBLASLT_MATMUL_TILE_16x16"
        if self == Self.CUBLASLT_MATMUL_TILE_32x8:
            return "CUBLASLT_MATMUL_TILE_32x8"
        if self == Self.CUBLASLT_MATMUL_TILE_8x64:
            return "CUBLASLT_MATMUL_TILE_8x64"
        if self == Self.CUBLASLT_MATMUL_TILE_16x32:
            return "CUBLASLT_MATMUL_TILE_16x32"
        if self == Self.CUBLASLT_MATMUL_TILE_32x16:
            return "CUBLASLT_MATMUL_TILE_32x16"
        if self == Self.CUBLASLT_MATMUL_TILE_64x8:
            return "CUBLASLT_MATMUL_TILE_64x8"
        if self == Self.CUBLASLT_MATMUL_TILE_32x32:
            return "CUBLASLT_MATMUL_TILE_32x32"
        if self == Self.CUBLASLT_MATMUL_TILE_32x64:
            return "CUBLASLT_MATMUL_TILE_32x64"
        if self == Self.CUBLASLT_MATMUL_TILE_64x32:
            return "CUBLASLT_MATMUL_TILE_64x32"
        if self == Self.CUBLASLT_MATMUL_TILE_32x128:
            return "CUBLASLT_MATMUL_TILE_32x128"
        if self == Self.CUBLASLT_MATMUL_TILE_64x64:
            return "CUBLASLT_MATMUL_TILE_64x64"
        if self == Self.CUBLASLT_MATMUL_TILE_128x32:
            return "CUBLASLT_MATMUL_TILE_128x32"
        if self == Self.CUBLASLT_MATMUL_TILE_64x128:
            return "CUBLASLT_MATMUL_TILE_64x128"
        if self == Self.CUBLASLT_MATMUL_TILE_128x64:
            return "CUBLASLT_MATMUL_TILE_128x64"
        if self == Self.CUBLASLT_MATMUL_TILE_64x256:
            return "CUBLASLT_MATMUL_TILE_64x256"
        if self == Self.CUBLASLT_MATMUL_TILE_128x128:
            return "CUBLASLT_MATMUL_TILE_128x128"
        if self == Self.CUBLASLT_MATMUL_TILE_256x64:
            return "CUBLASLT_MATMUL_TILE_256x64"
        if self == Self.CUBLASLT_MATMUL_TILE_64x512:
            return "CUBLASLT_MATMUL_TILE_64x512"
        if self == Self.CUBLASLT_MATMUL_TILE_128x256:
            return "CUBLASLT_MATMUL_TILE_128x256"
        if self == Self.CUBLASLT_MATMUL_TILE_256x128:
            return "CUBLASLT_MATMUL_TILE_256x128"
        if self == Self.CUBLASLT_MATMUL_TILE_512x64:
            return "CUBLASLT_MATMUL_TILE_512x64"
        if self == Self.CUBLASLT_MATMUL_TILE_64x96:
            return "CUBLASLT_MATMUL_TILE_64x96"
        if self == Self.CUBLASLT_MATMUL_TILE_96x64:
            return "CUBLASLT_MATMUL_TILE_96x64"
        if self == Self.CUBLASLT_MATMUL_TILE_96x128:
            return "CUBLASLT_MATMUL_TILE_96x128"
        if self == Self.CUBLASLT_MATMUL_TILE_128x160:
            return "CUBLASLT_MATMUL_TILE_128x160"
        if self == Self.CUBLASLT_MATMUL_TILE_160x128:
            return "CUBLASLT_MATMUL_TILE_160x128"
        if self == Self.CUBLASLT_MATMUL_TILE_192x128:
            return "CUBLASLT_MATMUL_TILE_192x128"
        if self == Self.CUBLASLT_MATMUL_TILE_128x192:
            return "CUBLASLT_MATMUL_TILE_128x192"
        if self == Self.CUBLASLT_MATMUL_TILE_128x96:
            return "CUBLASLT_MATMUL_TILE_128x96"
        if self == Self.CUBLASLT_MATMUL_TILE_32x256:
            return "CUBLASLT_MATMUL_TILE_32x256"
        if self == Self.CUBLASLT_MATMUL_TILE_256x32:
            return "CUBLASLT_MATMUL_TILE_256x32"
        if self == Self.CUBLASLT_MATMUL_TILE_END:
            return "CUBLASLT_MATMUL_TILE_END"
        return abort[String]("invalid cublasLtMatmulTile_t entry")

    fn __int__(self) raises -> Int:
        return int(self._value)


fn cublasLtGetStatusName(status: Result) raises -> Pointer[Int8]:
    return _get_dylib_function[
        "cublasLtGetStatusName", fn (Result) raises -> Pointer[Int8]
    ]()(status)


fn cublasLtMatmulPreferenceCreate(
    pref: Pointer[Pointer[cublasLtMatmulPreferenceOpaque_t]],
) raises -> Result:
    """Create new matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceCreate",
        fn (
            Pointer[Pointer[cublasLtMatmulPreferenceOpaque_t]],
        ) raises -> Result,
    ]()(pref)


@register_passable("trivial")
struct cublasLtMatmulHeuristicResult_t:
    """Results structure used by cublasLtMatmulGetAlgo.

    Holds returned configured algo descriptor and its runtime properties.
    ."""

    # Matmul algorithm descriptor.
    #
    # Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
    # CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID
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


fn cublasLtLoggerSetFile(file: Pointer[NoneType]) raises -> Result:
    """Experimental: Log file setter.

    file                         an open file with write permissions

    \retval     CUBLAS_STATUS_SUCCESS        if log file was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetFile", fn (Pointer[NoneType]) raises -> Result
    ]()(file)


fn cublasLtLoggerOpenFile(log_file: Pointer[Int8]) raises -> Result:
    """Experimental: Open log file.

    logFile                      log file path. if the log file does not exist, it will be created

    \retval     CUBLAS_STATUS_SUCCESS        if log file was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerOpenFile", fn (Pointer[Int8]) raises -> Result
    ]()(log_file)


fn cublasLtMatrixTransform(
    light_handle: Pointer[cublasLtContext],
    transform_desc: Pointer[cublasLtMatrixTransformDescOpaque_t],
    alpha: Pointer[NoneType],
    _a: Pointer[NoneType],
    _adesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    beta: Pointer[NoneType],
    _b: Pointer[NoneType],
    _bdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    _c: Pointer[NoneType],
    _cdesc: Pointer[cublasLtMatrixLayoutOpaque_t],
    stream: Pointer[Stream],
) raises -> Result:
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
            Pointer[cublasLtContext],
            Pointer[cublasLtMatrixTransformDescOpaque_t],
            Pointer[NoneType],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[NoneType],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[NoneType],
            Pointer[cublasLtMatrixLayoutOpaque_t],
            Pointer[Stream],
        ) raises -> Result,
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


fn cublasLtLoggerSetMask(mask: Int16) raises -> Result:
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
    return _get_dylib_function[
        "cublasLtLoggerSetMask", fn (Int16) raises -> Result
    ]()(mask)


# Opaque structure holding CUBLASLT context
# .
alias cublasLtHandle_t = Pointer[cublasLtContext]


fn cublasLtMatrixTransformDescGetAttribute(
    transform_desc: Pointer[cublasLtMatrixTransformDescOpaque_t],
    attr: cublasLtMatrixTransformDescAttributes_t,
    buf: Pointer[NoneType],
    size_in_bytes: Int,
    size_written: Pointer[Int],
) raises -> Result:
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
            Pointer[cublasLtMatrixTransformDescOpaque_t],
            cublasLtMatrixTransformDescAttributes_t,
            Pointer[NoneType],
            Int,
            Pointer[Int],
        ) raises -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes, size_written)


fn cublasLtMatmulDescInit_internal(
    matmul_desc: Pointer[cublasLtMatmulDescOpaque_t],
    size: Int,
    compute_type: ComputeType,
    scale_type: DataType,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescInit_internal",
        fn (
            Pointer[cublasLtMatmulDescOpaque_t],
            Int,
            ComputeType,
            DataType,
        ) raises -> Result,
    ]()(matmul_desc, size, compute_type, scale_type)


fn cublasLtMatmulPreferenceInit_internal(
    pref: Pointer[cublasLtMatmulPreferenceOpaque_t], size: Int
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceInit_internal",
        fn (Pointer[cublasLtMatmulPreferenceOpaque_t], Int) raises -> Result,
    ]()(pref, size)


@register_passable("trivial")
struct cublasLtMatrixTransformDescOpaque_t:
    """Semi-opaque descriptor for cublasLtMatrixTransform() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@value
@register_passable("trivial")
struct cublasLtMatrixTransformDescAttributes_t:
    """Matrix transform descriptor attributes to define details of the operation.
    ."""

    var _value: Int8
    alias CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = cublasLtMatrixTransformDescAttributes_t(
        0
    )
    """Scale type, see cudaDataType. Inputs are converted to scale type for scaling and summation and results are then
    converted to output type to store in memory.

    int32_t.
    """
    alias CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = cublasLtMatrixTransformDescAttributes_t(
        1
    )
    """Pointer mode of alpha and beta, see PointerMode.

    int32_t, default: HOST.
    """
    alias CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA = cublasLtMatrixTransformDescAttributes_t(
        2
    )
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    alias CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB = cublasLtMatrixTransformDescAttributes_t(
        3
    )
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) raises -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) raises -> Bool:
        return not (self == other)

    fn __str__(self) raises -> String:
        if self == Self.CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
            return "CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE"
        if self == Self.CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
            return "CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE"
        if self == Self.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
            return "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA"
        if self == Self.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
            return "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB"
        return abort[String](
            "invalid cublasLtMatrixTransformDescAttributes_t entry"
        )

    fn __int__(self) raises -> Int:
        return int(self._value)


# fn cublasLtMatmulAlgoGetIds(
#     light_handle: Pointer[cublasLtContext],
#     compute_type: ComputeType,
#     scale_type: DataType,
#     _atype: DataType,
#     _btype: DataType,
#     _ctype: DataType,
#     _dtype: DataType,
#     requested_algo_count: Int16,
#     algo_ids_array: UNKNOWN,
#     return_algo_count: Pointer[Int16],
# ) raises -> Result:
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
#             Pointer[cublasLtContext],
#             ComputeType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             Int16,
#             UNKNOWN,
#             Pointer[Int16],
#         ) raises -> Result,
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
