# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections.string.string_slice import StringSlice
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator, sizeof
from sys.ffi import OpaquePointer, _get_global_or_null, external_call

import gpu._rocblas
from buffer import DimList, NDBuffer
from gpu._cublas.cublas import (
    Algorithm,
    ComputeType,
    _convert_to_cublas_datatype,
    _convert_to_cublas_transpose,
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
    cublasGemmEx,
    cublasLoggerConfigure,
    cublasMath_t,
    cublasOperation_t,
    cublasSetMathMode,
    cublasSetStream,
)
from gpu._cublas.cublaslt import (
    Context,
    MatmulAlgorithm,
    Preference,
    cublasLtCreate,
    cublasLtDestroy,
    cublasLtGetVersion,
    cublasLtMatmul,
    cublasLtMatmulAlgoGetHeuristic,
    cublasLtMatmulAlgoInit,
    cublasLtMatmulDesc_t,
    cublasLtMatmulDescAttributes_t,
    cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy,
)
from gpu._cublas.dtype import DataType
from gpu._cublas.result import Result
from gpu._rocblas.hipblaslt import (
    _check_hipblas_error,
    _convert_to_hip_datatype,
    hipblasComputeType_t,
    hipblasLtCreate,
    hipblasLtDestroy,
    hipblasLtHandle_t,
    hipblasLtMatmul,
    hipblasLtMatmulAlgoGetHeuristic,
    hipblasLtMatmulDesc_t,
    hipblasLtMatmulDescAttributes_t,
    hipblasLtMatmulDescCreate,
    hipblasLtMatmulDescDestroy,
    hipblasLtMatmulDescSetAttribute,
    hipblasLtMatmulHeuristicResult_t,
    hipblasLtMatmulPreference_t,
    hipblasLtMatmulPreferenceCreate,
    hipblasLtMatmulPreferenceDestroy,
    hipblasLtMatrixLayout_t,
    hipblasLtMatrixLayoutCreate,
    hipblasLtMatrixLayoutDestroy,
    hipblasOperation_t,
    hipDataType_t,
)
from gpu.host import DeviceContext
from gpu.host._amdgpu_hip import HIP
from gpu.host._nvidia_cuda import CUDA
from layout import Layout
from memory import UnsafePointer

from utils.variant import Variant
from gpu.host.info import DEFAULT_GPU, H100, Vendor

# ===----------------------------------------------------------------------===#
# Backend
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Backend:
    var _value: Int32

    alias AUTOMATIC = Self(0)
    alias CUBLAS = Self(1)
    alias CUBLASLT = Self(2)
    alias ROCBLAS = Self(3)
    alias HIPBLASLT = Self(4)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.AUTOMATIC:
            return writer.write("AUTOMATIC")
        if self is Self.CUBLAS:
            return writer.write("CUBLAS")
        if self is Self.CUBLASLT:
            return writer.write("CUBLASLT")
        if self is Self.ROCBLAS:
            return writer.write("ROCBLAS")
        writer.write("HIPBLASLT")


fn _resolve_backend[backend: Backend, type: DType = DType.invalid]() -> Backend:
    @parameter
    if backend is not Backend.AUTOMATIC:
        return backend
    elif has_amd_gpu_accelerator():
        return Backend.ROCBLAS
    elif type.is_float8() or (
        DEFAULT_GPU.vendor == Vendor.NVIDIA_GPU and DEFAULT_GPU > H100
    ):
        return Backend.CUBLASLT
    return Backend.CUBLAS


# ===----------------------------------------------------------------------===#
# Handle
# ===----------------------------------------------------------------------===#


@value
struct Handle[backend: Backend = _resolve_backend[Backend.AUTOMATIC]()]:
    alias resolved_backend = _resolve_backend[backend]()
    alias _cublas_type = UnsafePointer[cublasContext]
    alias _cublaslt_type = UnsafePointer[Context]
    alias _rocblas_type = _rocblas.Handle
    alias _hipblaslt_type = hipblasLtHandle_t
    alias type = Variant[
        Self._cublas_type, Self._cublaslt_type, Self._rocblas_type
    ]
    var _handle: Self.type

    fn __init__(out self) raises:
        @parameter
        if Self.resolved_backend is Backend.CUBLAS:
            var handle = Self._cublas_type()
            check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
            self._handle = handle
        elif Self.resolved_backend is Backend.CUBLASLT:
            var handle = Self._cublaslt_type()
            check_cublas_error(cublasLtCreate(UnsafePointer.address_of(handle)))
            self._handle = handle
        elif Self.resolved_backend is Backend.ROCBLAS:
            var handle = Self._rocblas_type()
            _rocblas.check_error(
                _rocblas.rocblas.rocblas_create_handle(
                    UnsafePointer.address_of(handle)
                )
            )
            self._handle = handle
        elif Self.resolved_backend is Backend.HIPBLASLT:
            var handle = Self._hipblaslt_type()
            _check_hipblas_error(
                hipblasLtCreate(UnsafePointer.address_of(handle))
            )
            self._handle = handle
        else:
            raise Error(
                "the backend '",
                backend,
                "' is not currently supported",
            )

    @always_inline
    fn __enter__(self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self) raises:
        @parameter
        if Self.resolved_backend is Backend.CUBLAS:
            check_cublas_error(cublasDestroy(self._get_cublas()))
            self._handle = Self._cublas_type()
            return
        elif Self.resolved_backend is Backend.CUBLASLT:
            check_cublas_error(cublasLtDestroy(self._get_cublaslt()))
            self._handle = Self._cublaslt_type()
            return
        elif Self.resolved_backend is Backend.ROCBLAS:
            _rocblas.check_error(
                _rocblas.rocblas.rocblas_destroy_handle(self._get_rocblas())
            )
            self._handle = Self._rocblas_type()
            return
        elif Self.resolved_backend is Backend.HIPBLASLT:
            _check_hipblas_error(hipblasLtDestroy(self._get_hipblaslt()))
            self._handle = Self._hipblaslt_type()
            return

        raise Error("the backend is not currently supported")

    fn _is_null(self) -> Bool:
        @parameter
        if Self.resolved_backend is Backend.CUBLAS:
            return self._get_cublas() == Self._cublas_type()
        elif Self.resolved_backend is Backend.CUBLASLT:
            return self._get_cublaslt() == Self._cublaslt_type()
        elif Self.resolved_backend is Backend.ROCBLAS:
            return self._get_rocblas() == Self._rocblas_type()
        elif Self.resolved_backend is Backend.HIPBLASLT:
            return self._get_hipblaslt() == Self._hipblaslt_type()

        return False

    fn _get_cublas(self) -> Self._cublas_type:
        constrained[
            Self.resolved_backend is Backend.CUBLAS, "backend must be CUBLAS"
        ]()
        return self._handle[Self._cublas_type]

    fn _get_cublaslt(self) -> Self._cublas_type:
        constrained[
            Self.resolved_backend is Backend.CUBLASLT,
            "backend must be CUBLASLT",
        ]()
        return self._handle[Self._cublaslt_type]

    fn _get_rocblas(self) -> Self._rocblas_type:
        constrained[
            Self.resolved_backend is Backend.ROCBLAS, "backend must be ROCBLAS"
        ]()
        return self._handle[Self._rocblas_type]

    fn _get_hipblaslt(self) -> Self._hipblaslt_type:
        constrained[
            Self.resolved_backend is Backend.HIPBLASLT,
            "backend must be HIPBLASLT",
        ]()
        return self._handle[Self._hipblaslt_type]

    fn __is__(self, other: Backend) -> Bool:
        return Self.resolved_backend is other

    fn __isnot__(self, other: Backend) -> Bool:
        return Self.resolved_backend is not other


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#

alias _DEBUG_CUBLAS = False


fn _attach_handle_to_stream(ctx: DeviceContext, handle: Handle) raises:
    @parameter
    if handle.resolved_backend is Backend.CUBLAS:
        check_cublas_error(
            cublasSetStream(handle._get_cublas(), CUDA(ctx.stream()))
        )

        @parameter
        if _DEBUG_CUBLAS:
            check_cublas_error(
                cublasLoggerConfigure(1, 1, 0, UnsafePointer[Int8]())
            )


fn _get_global_handle[
    backend: Backend = _resolve_backend[Backend.AUTOMATIC]()
](ctx: DeviceContext) raises -> Handle[backend]:
    alias HANDLE_NAME = String("LINALG_VENDOR_BLAS_", backend)
    if global_ptr := _get_global_or_null[HANDLE_NAME]().bitcast[
        Handle[backend]
    ]():
        _attach_handle_to_stream(ctx, global_ptr[])
        return global_ptr[]

    # Otherwise, we have not initialized the handle yet.
    var handle_ptr = UnsafePointer[Handle[backend]].alloc(1)
    handle_ptr.init_pointee_move(Handle[backend]())
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(HANDLE_NAME),
        handle_ptr.bitcast[NoneType](),
    )

    _attach_handle_to_stream(ctx, handle_ptr[])

    return handle_ptr[]


fn matmul[
    use_tf32: Bool = False
](
    ctx: DeviceContext,
    c: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    """
    Matmul using the vendor BLAS library. With a global handle.
    """

    return matmul[use_tf32](
        ctx,
        _get_global_handle(ctx),
        c,
        a,
        b,
        c_row_major=c_row_major,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        alpha=alpha,
        beta=beta,
    )


fn matmul[
    use_tf32: Bool = False
](
    ctx: DeviceContext,
    handle: Handle,
    c: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    @parameter
    if handle.resolved_backend is Backend.CUBLAS:
        _cublas_matmul[use_tf32=use_tf32](
            ctx,
            handle._get_cublas(),
            c,
            a,
            b,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
        )
    elif handle.resolved_backend is Backend.ROCBLAS:
        _rocblas_matmul[use_tf32=use_tf32](
            ctx,
            handle._get_rocblas(),
            c,
            a,
            b,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
        )
    elif handle.resolved_backend is Backend.CUBLASLT:
        _cublasLt_matmul(
            ctx,
            handle._get_cublaslt(),
            c,
            a,
            b,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
        )
    elif handle.resolved_backend is Backend.HIPBLASLT:
        _hipblasLt_matmul(
            ctx,
            handle._get_hipblaslt(),
            c,
            a,
            b,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise String(
            "the backend '",
            handle.backend,
            "' is not currently supported",
        )


# ===----------------------------------------------------------------------===#
# CUBLAS
# ===----------------------------------------------------------------------===#


fn _cublas_matmul[
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: UnsafePointer[cublasContext],
    c: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    constrained[
        a.type == b.type
        and (a.type is DType.float32 or a.type.is_half_float()),
        (
            "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
            " it if more types are needed."
        ),
    ]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]() if not transpose_a else a.dim[0]()

    var compute_type: ComputeType

    @parameter
    if a.type is DType.float16:
        compute_type = ComputeType.COMPUTE_32F
    elif a.type is DType.bfloat16:
        compute_type = ComputeType.COMPUTE_32F
    else:
        compute_type = (
            ComputeType.COMPUTE_32F_FAST_TF32 if use_tf32 else ComputeType.COMPUTE_32F
        )

    # When use_tf32 is True, CUBLAS will use TF32 to speedup the computation.
    # However, the result is not bit-wise identical to the result of FP32.
    @parameter
    if use_tf32:
        check_cublas_error(
            cublasSetMathMode(handle, cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH)
        )
    else:
        check_cublas_error(
            cublasSetMathMode(handle, cublasMath_t.CUBLAS_DEFAULT_MATH)
        )

    # Rocblas is by default column-major but we like to have the output in row-major
    # to compare with our results. To do this without an explicit transpose, we
    # can swap A, B and output a NxM column-major matrix, which is same as
    # MxN row-major i.e.
    #
    #      C: MxN_row_major = A: MxK_row_major @ B: KxN_row_major
    #   => C: NxM_col_major = B: NxK_col_major @ A: KxM_col_major
    #
    # I haven't seen any significant performance difference before and after this
    # transformation. To be rigorous though, we should set `c_is_row_major = True`
    # for accuracy validations and uses default column-major in benchmark.

    if c_row_major:
        return check_cublas_error(
            cublasGemmEx(
                handle,
                _convert_to_cublas_transpose(transpose_b),
                _convert_to_cublas_transpose(transpose_a),
                N,
                M,
                K,
                UnsafePointer.address_of(alpha).bitcast[NoneType](),
                UnsafePointer(b.data.bitcast[NoneType]()),
                _convert_to_cublas_datatype[b.type](),
                K if transpose_b else N,
                UnsafePointer(a.data.bitcast[NoneType]()),
                _convert_to_cublas_datatype[a.type](),
                K,
                UnsafePointer.address_of(beta).bitcast[NoneType](),
                UnsafePointer(c.data.bitcast[NoneType]()),
                _convert_to_cublas_datatype[c.type](),
                N,
                compute_type,
                Algorithm.DEFAULT,
            ),
            msg=String(
                "failed to operate on cublas on the shape C=",
                c.dynamic_shape,
                "x",
                c.type,
                ", A=",
                a.dynamic_shape,
                "x",
                a.type,
                ", B=",
                b.dynamic_shape,
                "x",
                b.type,
            ),
        )
    # Default column-major.
    check_cublas_error(
        cublasGemmEx(
            handle,
            _convert_to_cublas_transpose(transpose_a),
            _convert_to_cublas_transpose(transpose_b),
            M,
            N,
            K,
            UnsafePointer.address_of(alpha).bitcast[NoneType](),
            UnsafePointer(a.data.bitcast[NoneType]()),
            _convert_to_cublas_datatype[a.type](),
            M,
            UnsafePointer(b.data.bitcast[NoneType]()),
            _convert_to_cublas_datatype[b.type](),
            N if transpose_b else K,
            UnsafePointer.address_of(beta).bitcast[NoneType](),
            UnsafePointer(c.data.bitcast[NoneType]()),
            _convert_to_cublas_datatype[c.type](),
            M,
            compute_type,
            Algorithm.DEFAULT,
        ),
        msg=String(
            "failed to operate on cublas on the shape C=",
            c.dynamic_shape,
            "x",
            c.type,
            ", A=",
            a.dynamic_shape,
            "x",
            a.type,
            ", B=",
            b.dynamic_shape,
            "x",
            b.type,
        ),
    )


# ===----------------------------------------------------------------------===#
# ROCBLAS
# ===----------------------------------------------------------------------===#


fn _rocblas_matmul[
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: _rocblas.Handle,
    c: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    constrained[
        a.type == b.type
        and (a.type is DType.float32 or a.type.is_half_float()),
        (
            "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
            " it if more types are needed."
        ),
    ]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]() if not transpose_a else a.dim[0]()

    var compute_type = _rocblas.types.DataType(DType.float32)

    # Cublas is by default column-major but we like to have the output in row-major
    # to compare with our results. To do this without an explicit transpose, we
    # can swap A, B and output a NxM column-major matrix, which is same as
    # MxN row-major i.e.
    #
    #      C: MxN_row_major = A: MxK_row_major @ B: KxN_row_major
    #   => C: NxM_col_major = B: NxK_col_major @ A: KxM_col_major
    #
    # I haven't seen any significant performance difference before and after this
    # transformation. To be rigorous though, we should set `c_is_row_major = True`
    # for accuracy validations and uses default column-major in benchmark.

    fn _convert_to_rocblas_transpose(tr: Bool) -> _rocblas.types.Operation:
        if tr:
            return _rocblas.types.Operation.TRANSPOSE
        return _rocblas.types.Operation.NONE

    if c_row_major:
        return _rocblas.check_error(
            _rocblas.rocblas.rocblas_gemm_ex(
                handle,
                _convert_to_rocblas_transpose(transpose_b),
                _convert_to_rocblas_transpose(transpose_a),
                N,
                M,
                K,
                UnsafePointer.address_of(alpha).bitcast[NoneType](),
                UnsafePointer(b.data.bitcast[NoneType]()),
                _rocblas.types.DataType(b.type),
                K if transpose_b else N,
                UnsafePointer(a.data.bitcast[NoneType]()),
                _rocblas.types.DataType(a.type),
                K,
                UnsafePointer.address_of(beta).bitcast[NoneType](),
                UnsafePointer(c.data.bitcast[NoneType]()),
                _rocblas.types.DataType(c.type),
                N,
                UnsafePointer(c.data.bitcast[NoneType]()),
                _rocblas.types.DataType(c.type),
                N,
                compute_type,
                _rocblas.rocblas.types.Algorithm.STANDARD,
                0,
                0,
            )
        )
    # Default column-major.
    _rocblas.check_error(
        _rocblas.rocblas.rocblas_gemm_ex(
            handle,
            _convert_to_rocblas_transpose(transpose_a),
            _convert_to_rocblas_transpose(transpose_b),
            M,
            N,
            K,
            UnsafePointer.address_of(alpha).bitcast[NoneType](),
            UnsafePointer(a.data.bitcast[NoneType]()),
            _rocblas.types.DataType(a.type),
            M,
            UnsafePointer(b.data.bitcast[NoneType]()),
            _rocblas.types.DataType(b.type),
            N if transpose_b else K,
            UnsafePointer.address_of(beta).bitcast[NoneType](),
            UnsafePointer(c.data.bitcast[NoneType]()),
            _rocblas.types.DataType(c.type),
            M,
            UnsafePointer(c.data.bitcast[NoneType]()),
            _rocblas.types.DataType(c.type),
            M,
            compute_type,
            _rocblas.rocblas.types.Algorithm.STANDARD,
            0,
            0,
        )
    )


# ===----------------------------------------------------------------------===#
# CUBLASLT
# ===----------------------------------------------------------------------===#


fn _cublasLt_matmul(
    ctx: DeviceContext,
    handle: UnsafePointer[Context],
    d: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    alias a_type = a.type
    alias b_type = b.type
    alias d_type = d.type
    var M = d.dim[0]()
    var N = d.dim[1]()
    var K = a.dim[1]()

    constrained[
        (
            a_type
            in (
                DType.float8_e4m3fn,
                DType.float8_e5m2,
                DType.bfloat16,
                DType.float16,
            )
        ),
        (
            "Only E4M3, E5M2, bfloat16, and float16 input data types are"
            " supported. Please extend it if you need more data types."
        ),
    ]()

    constrained[a_type == b_type, "A and B must have the same type"]()

    @parameter
    if a_type.is_float8():
        constrained[
            not (a_type == b_type == DType.float8_e5m2),
            (
                "E5M2xE5m2 is not supported! Please refer to"
                " `https://docs.nvidia.com/cuda/cublas/#id105`"
            ),
        ]()

    if transpose_a or not transpose_b:
        raise Error(
            "the cuBLASLT backend currently only is implemented for"
            " transpose_a=False and transpose_b=True"
        )

    var cuda_stream = CUDA(ctx.stream())

    # CublasLt is by default column-major but we like to have the output in row-major
    # to compare with our results. Use `c_row_major` to determine the output layout.

    # To use FP8 kernels, the following set of requirements must be satisfied:
    # 1) All matrix dimensions must meet the optimal requirements listed in Tensor Core Usage (See Below)
    # 2) A must be transposed and B non-transposed (The “TN” format).
    # 3) The compute type must be CUBLAS_COMPUTE_32F.
    # 4) The scale type must be CUDA_R_32F.

    # A verity of A, B, and D data types are supported by this API. For more
    # information please refer to `https://docs.nvidia.com/cuda/cublas/#id105`

    # The best performance when using Tensor Cores can be achieved when the matrix dimensions and
    # pointers meet certain memory alignment requirements.
    # Specifically, all of the following conditions must be satisfied to get the most performance out of Tensor Cores:
    # 1) ((op_A == CUBLAS_OP_N ? m : k) * AtypeSize) % 16 == 0
    # 2) ((op_B == CUBLAS_OP_N ? k : n) * BtypeSize) % 16 == 0
    # 3) (m * CtypeSize) % 16 == 0
    # 4) (lda * AtypeSize) % 16 == 0
    # 5) (ldb * BtypeSize) % 16 == 0
    # 6) (ldc * CtypeSize) % 16 == 0
    # 7) intptr_t(A) % 16 == 0
    # 8) intptr_t(B) % 16 == 0
    # 9) intptr_t(C) % 16 == 0

    # TN format required for FP8
    var transa = cublasOperation_t.CUBLAS_OP_T
    var transb = cublasOperation_t.CUBLAS_OP_N

    # create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    var compute_desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer.address_of(compute_desc),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        )
    )

    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA,
            UnsafePointer.address_of(transa).bitcast[NoneType](),
            sizeof[cublasOperation_t](),
        )
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB,
            UnsafePointer.address_of(transb).bitcast[NoneType](),
            sizeof[cublasOperation_t](),
        )
    )

    # create matrix descriptors, we are good with the details here so no need to set any extra attributes
    # table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    var _adesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer.address_of(_adesc),
            _convert_to_cublas_datatype[a_type](),
            K,
            N if c_row_major else M,
            K,
        )
    )

    var _bdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer.address_of(_bdesc),
            _convert_to_cublas_datatype[b_type](),
            K,
            M if c_row_major else N,
            K,
        )
    )

    var _ddesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer.address_of(_ddesc),
            _convert_to_cublas_datatype[d_type](),
            N if c_row_major else M,
            M if c_row_major else N,
            N if c_row_major else M,
        )
    )

    var _cdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer.address_of(_cdesc),
            _convert_to_cublas_datatype[d_type](),
            N if c_row_major else M,
            M if c_row_major else N,
            N if c_row_major else M,
        )
    )

    var preference = cublasLtMatmulPreference_t()
    check_cublas_error(
        cublasLtMatmulPreferenceCreate(UnsafePointer.address_of(preference))
    )

    var workspace_size = 32 * 1024 * 1024
    check_cublas_error(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            Preference.MAX_WORKSPACE_BYTES,
            UnsafePointer.address_of(workspace_size).bitcast[NoneType](),
            sizeof[Int64](),
        )
    )

    var heuristic_result = cublasLtMatmulHeuristicResult_t()
    var returned_results = 0
    check_cublas_error(
        cublasLtMatmulAlgoGetHeuristic(
            handle,
            compute_desc,
            _adesc,
            _bdesc,
            _cdesc,
            _ddesc,
            preference,
            1,
            UnsafePointer.address_of(heuristic_result),
            UnsafePointer.address_of(returned_results),
        )
    )

    if returned_results == 0:
        raise Error("No algorithm was found!")

    var matmul_workspace = ctx.enqueue_create_buffer[DType.uint8](
        workspace_size
    )

    if c_row_major:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                UnsafePointer.address_of(alpha).bitcast[NoneType](),  # alpha
                UnsafePointer(b.data.bitcast[NoneType]()),  # _a
                _adesc,  # _adesc
                UnsafePointer(a.data.bitcast[NoneType]()),  # _b
                _bdesc,  # _bdesc
                UnsafePointer.address_of(beta).bitcast[NoneType](),  # beta
                UnsafePointer[NoneType](),  # _c
                _cdesc,  # _cdesc
                UnsafePointer(d.data.bitcast[NoneType]()),  # _d
                _ddesc,  # _ddesc
                UnsafePointer.address_of(heuristic_result.algo),  # algo
                UnsafePointer(
                    matmul_workspace.unsafe_ptr().bitcast[NoneType]()
                ),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream[],  # stream
            )
        )
    else:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                UnsafePointer.address_of(alpha).bitcast[NoneType](),  # alpha
                UnsafePointer(a.data.bitcast[NoneType]()),  # _a
                _adesc,  # _adesc
                UnsafePointer(b.data.bitcast[NoneType]()),  # _b
                _bdesc,  # _bdesc
                UnsafePointer.address_of(beta).bitcast[NoneType](),  # beta
                UnsafePointer[NoneType](),  # _c
                _cdesc,  # _cdesc
                UnsafePointer(d.data.bitcast[NoneType]()),  # _d
                _ddesc,  # _ddesc
                UnsafePointer.address_of(heuristic_result.algo),  # algo
                UnsafePointer(
                    matmul_workspace.unsafe_ptr().bitcast[NoneType]()
                ),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream[],  # stream
            )
        )

    check_cublas_error(cublasLtMatmulDescDestroy(compute_desc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_adesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_bdesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_cdesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_ddesc))
    check_cublas_error(cublasLtMatmulPreferenceDestroy(preference))

    _ = matmul_workspace^


# ===----------------------------------------------------------------------===#
# HIPBLASLT
# ===----------------------------------------------------------------------===#


fn _hipblasLt_matmul(
    ctx: DeviceContext,
    handle: hipblasLtHandle_t,
    d: NDBuffer[_, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    *,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    constrained[
        (
            a.type in [DType.float32, DType.float16, DType.bfloat16]
            and b.type in [DType.float32, DType.float16, DType.bfloat16]
        ),
        (
            "Only FP32, FP16, BF16 input data types are supported. Please"
            " extend it if you need more data types."
        ),
    ]()

    @always_inline
    @parameter
    fn create_matrix_layout(
        buf: NDBuffer[_, 2, _, _]
    ) raises -> hipblasLtMatrixLayout_t:
        var _desc = hipblasLtMatrixLayout_t()
        _check_hipblas_error(
            hipblasLtMatrixLayoutCreate(
                UnsafePointer.address_of(_desc),
                _convert_to_hip_datatype[buf.type](),
                buf.dim[1](),
                buf.dim[0](),
                buf.dim[1](),
            )
        )
        return _desc

    var _adata = UnsafePointer(a.data.bitcast[NoneType]())
    var _bdata = UnsafePointer(b.data.bitcast[NoneType]())
    var _ddata = UnsafePointer(d.data.bitcast[NoneType]())

    var _adesc = create_matrix_layout(a)
    var _bdesc = create_matrix_layout(b)
    var _ddesc = create_matrix_layout(d)

    var transa = hipblasOperation_t.OP_T if transpose_a else hipblasOperation_t.OP_N
    var transb = hipblasOperation_t.OP_T if transpose_b else hipblasOperation_t.OP_N

    # hipblasLt is by default column-major but we like to have the output in row-major
    # to compare with our results. Use `c_row_major` to determine the output layout.
    if c_row_major:
        swap(_adata, _bdata)
        swap(_adesc, _bdesc)
        swap(transa, transb)

    var operationDesc = hipblasLtMatmulDesc_t()
    _check_hipblas_error(
        hipblasLtMatmulDescCreate(
            UnsafePointer.address_of(operationDesc),
            hipblasComputeType_t.COMPUTE_32F,
            hipDataType_t.R_32F,
        )
    )

    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSA,
            UnsafePointer.address_of(transa).bitcast[NoneType](),
            sizeof[hipblasOperation_t](),
        )
    )
    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSB,
            UnsafePointer.address_of(transb).bitcast[NoneType](),
            sizeof[hipblasOperation_t](),
        )
    )

    var preference = hipblasLtMatmulPreference_t()
    _check_hipblas_error(
        hipblasLtMatmulPreferenceCreate(UnsafePointer.address_of(preference))
    )

    var heuristicResult = hipblasLtMatmulHeuristicResult_t()
    var returnedResults = 0
    _check_hipblas_error(
        hipblasLtMatmulAlgoGetHeuristic(
            handle,
            operationDesc,
            _adesc,
            _bdesc,
            _ddesc,
            _ddesc,
            preference,
            1,
            UnsafePointer.address_of(heuristicResult),
            UnsafePointer.address_of(returnedResults),
        )
    )

    if returnedResults == 0:
        raise Error("No algorithm was found!")

    _check_hipblas_error(
        hipblasLtMatmul(
            handle,  # light_handle
            operationDesc,  # compute_desc
            UnsafePointer.address_of(alpha).bitcast[NoneType](),  # alpha
            _adata,  # _a
            _adesc,  # _adesc
            _bdata,  # _b
            _bdesc,  # _bdesc
            UnsafePointer.address_of(beta).bitcast[NoneType](),  # beta
            _ddata,  # _c
            _ddesc,  # _cdesc
            _ddata,  # _d
            _ddesc,  # _ddesc
            UnsafePointer.address_of(heuristicResult.algo),  # algo
            UnsafePointer[NoneType](),  # workspace
            0,  # workspace_size_in_bytes
            HIP(ctx.stream())[],  # stream
        )
    )

    _check_hipblas_error(hipblasLtMatmulPreferenceDestroy(preference))
    _check_hipblas_error(hipblasLtMatmulDescDestroy(operationDesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_adesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_bdesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_ddesc))
