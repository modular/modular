# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]

from sys import has_amd_gpu_accelerator, size_of
from ffi import _get_global_or_null, external_call

import _rocblas
from _cublas.cublas import (
    Algorithm,
    ComputeType,
    _convert_to_cublas_datatype,
    _convert_to_cublas_transpose,
    check_cublas_error,
    cublasHandle_t,
    cublasCreate,
    cublasDestroy,
    cublasGemmEx,
    cublasLoggerConfigure,
    cublasMath_t,
    cublasOperation_t,
    cublasSetMathMode,
    cublasSetStream,
)
from _cublas.cublaslt import (
    Context,
    Preference,
    cublasLtLoggerSetLevel,
    cublasLtMatmul,
    cublasLtMatmulAlgoGetHeuristic,
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
    cublasLtMatmulMatrixScale_t,
)
from _cublas.dtype import DataType
from _rocblas.hipblaslt import (
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
    hipblasLtMatrixLayoutSetAttribute,
    hipblasLtMatmulLayoutAttribute_t,
    hipblasOperation_t,
    hipDataType_t,
)
from buffer import NDBuffer
from gpu.host import DeviceContext
from gpu.host._amdgpu_hip import HIP
from gpu.host._nvidia_cuda import CUDA
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout._ndbuffer_stub import from_ndbuffer_row_major
from runtime.tracing import Trace, TraceLevel
from buffer import DimList, NDBuffer
from utils import IndexList
from utils.variant import Variant
from gpu.host.info import B200
from collections import OptionalReg, Optional
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
)

# ===----------------------------------------------------------------------===#
# Backend
# ===----------------------------------------------------------------------===#


struct Backend(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32

    comptime AUTOMATIC = Self(0)
    comptime CUBLAS = Self(1)
    comptime CUBLASLT = Self(2)
    comptime ROCBLAS = Self(3)
    comptime HIPBLASLT = Self(4)

    fn __init__(out self, value: Int):
        self._value = Int32(value)

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

    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.AUTOMATIC:
            return writer.write("AUTOMATIC")
        if self is Self.CUBLAS:
            return writer.write("CUBLAS")
        if self is Self.CUBLASLT:
            return writer.write("CUBLASLT")
        if self is Self.ROCBLAS:
            return writer.write("ROCBLAS")
        writer.write("HIPBLASLT")


fn _resolve_backend[
    backend: Backend, dtype: DType = DType.invalid
]() -> Backend:
    comptime if backend is not Backend.AUTOMATIC:
        return backend
    # TODO: Remove this once we have a proper hipBLASLt backend for float32.
    elif dtype == DType.float32 and has_amd_gpu_accelerator():
        return Backend.ROCBLAS
    elif has_amd_gpu_accelerator():
        return Backend.HIPBLASLT
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    elif dtype.is_float8() or dtype == DType.uint8:
        return Backend.CUBLASLT
    return Backend.CUBLAS


# ===----------------------------------------------------------------------===#
# Handle
# ===----------------------------------------------------------------------===#


struct Handle[backend: Backend = _resolve_backend[Backend.AUTOMATIC]()](
    ImplicitlyCopyable
):
    comptime resolved_backend = _resolve_backend[Self.backend]()
    comptime _cublas_type = cublasHandle_t
    comptime _rocblas_type = _rocblas.Handle
    comptime _hipblaslt_type = hipblasLtHandle_t
    comptime type = Variant[
        Self._cublas_type,
        Self._rocblas_type,
        Self._hipblaslt_type,
    ]
    var _handle: Self.type

    fn __init__(out self) raises:
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            var handle = Self._cublas_type()
            check_cublas_error(
                cublasCreate(LegacyUnsafePointer(to=handle).as_unsafe_pointer())
            )
            self._handle = handle
        elif Self.resolved_backend is Backend.ROCBLAS:
            var handle = Self._rocblas_type()
            _rocblas.check_error(
                _rocblas.rocblas.rocblas_create_handle(
                    LegacyUnsafePointer(to=handle)
                )
            )
            self._handle = handle
        elif Self.resolved_backend is Backend.HIPBLASLT:
            var handle = Self._hipblaslt_type()
            _check_hipblas_error(
                hipblasLtCreate(LegacyUnsafePointer(to=handle))
            )
            self._handle = handle
        else:
            raise Error(
                "the backend '",
                Self.backend,
                "' is not currently supported",
            )

    @always_inline
    fn __enter__(self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self) raises:
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            check_cublas_error(cublasDestroy(self._get_cublas()))
            self._handle = Self._cublas_type()
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
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            return self._get_cublas() == Self._cublas_type()
        elif Self.resolved_backend is Backend.ROCBLAS:
            return self._get_rocblas() == Self._rocblas_type()
        elif Self.resolved_backend is Backend.HIPBLASLT:
            return self._get_hipblaslt() == Self._hipblaslt_type()

        return False

    fn _get_cublas(self) -> Self._cublas_type:
        comptime assert Self.resolved_backend in (
            Backend.CUBLAS,
            Backend.CUBLASLT,
        ), "backend must be CUBLAS/CUBLASLT"
        return self._handle[Self._cublas_type]

    fn _get_rocblas(self) -> Self._rocblas_type:
        comptime assert (
            Self.resolved_backend is Backend.ROCBLAS
        ), "backend must be ROCBLAS"
        return self._handle[Self._rocblas_type]

    fn _get_hipblaslt(self) -> Self._hipblaslt_type:
        comptime assert (
            Self.resolved_backend is Backend.HIPBLASLT
        ), "backend must be HIPBLASLT"
        return self._handle[Self._hipblaslt_type]

    fn __is__(self, other: Backend) -> Bool:
        return Self.resolved_backend is other

    fn __isnot__(self, other: Backend) -> Bool:
        return Self.resolved_backend is not other


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#

comptime _DEBUG_VENDOR_BLAS = False


fn _attach_handle_to_stream(ctx: DeviceContext, handle: Handle) raises:
    comptime if handle.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
        check_cublas_error(
            cublasSetStream(handle._get_cublas(), CUDA(ctx.stream()))
        )

        comptime if _DEBUG_VENDOR_BLAS:
            comptime if handle.resolved_backend is Backend.CUBLAS:
                check_cublas_error(
                    cublasLoggerConfigure(1, 1, 0, UnsafePointer[Int8]())
                )
            else:
                check_cublas_error(cublasLtLoggerSetLevel(5))

    elif handle.resolved_backend is Backend.ROCBLAS:
        _rocblas.check_error(
            _rocblas.rocblas.rocblas_set_stream(
                handle._get_rocblas(), HIP(ctx.stream())
            )
        )


fn _get_global_handle[
    dtype: DType,
    backend: Backend = _resolve_backend[Backend.AUTOMATIC, dtype=dtype](),
](ctx: DeviceContext) raises -> Handle[backend]:
    var HANDLE_NAME = String("LINALG_VENDOR_BLAS_", backend, "_", ctx.id())
    if global_ptr := _get_global_or_null(HANDLE_NAME).bitcast[
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
    use_tf32: Bool = False,
    *,
    scales_type: DType = DType.invalid,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[mut=False, _, 2, _, _],
    b: NDBuffer[mut=False, _, 2, _, _],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    """
    Matmul using the vendor BLAS library. With a global handle.
    """

    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)

    # Push the device context to ensure correct CUDA context is current for all
    # vendor BLAS calls.
    with ctx.push_context() as cur_ctx:
        return matmul[use_tf32=use_tf32, scales_type=scales_type](
            cur_ctx,
            _get_global_handle[a.type](ctx),
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales=a_scales,
            b_scales=b_scales,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
        )


fn matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    *,
    use_tf32: Bool = False,
    scales_type: DType = DType.invalid,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    c_tensor: LayoutTensor[c_type, c_layout, _],
    a_tensor: LayoutTensor[a_type, a_layout, _],
    b_tensor: LayoutTensor[b_type, b_layout, _],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    with ctx.push_context() as cur_ctx:
        return matmul[use_tf32=use_tf32, scales_type=scales_type](
            cur_ctx,
            _get_global_handle[a_type](ctx),
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales=a_scales,
            b_scales=b_scales,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
        )


fn matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    use_tf32: Bool = False,
    scales_type: DType = DType.invalid,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    handle: Handle,
    c_tensor: LayoutTensor[c_type, c_layout, _],
    a_tensor: LayoutTensor[a_type, a_layout, _],
    b_tensor: LayoutTensor[b_type, b_layout, _],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    comptime if handle.resolved_backend is Backend.CUBLAS:
        with Trace[TraceLevel.OP]("_cublas_matmul"):
            _cublas_matmul[use_tf32=use_tf32](
                ctx,
                handle._get_cublas(),
                c_tensor,
                a_tensor,
                b_tensor,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
    elif handle.resolved_backend is Backend.ROCBLAS:
        with Trace[TraceLevel.OP]("_rocblas_matmul"):
            _rocblas_matmul[use_tf32=use_tf32](
                ctx,
                handle._get_rocblas(),
                c_tensor,
                a_tensor,
                b_tensor,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
    elif handle.resolved_backend is Backend.CUBLASLT:
        with Trace[TraceLevel.OP]("_cublasLt_matmul"):
            _cublasLt_matmul(
                ctx,
                handle._get_cublas().bitcast[Context](),
                c_tensor,
                a_tensor,
                b_tensor,
                a_scales=a_scales,
                b_scales=b_scales,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
    elif handle.resolved_backend is Backend.HIPBLASLT:
        with Trace[TraceLevel.OP]("_hipblasLt_matmul"):
            _hipblasLt_matmul(
                ctx,
                handle._get_hipblaslt(),
                c_tensor,
                a_tensor,
                b_tensor,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
                batch_size=batch_size,
            )
    else:
        raise Error(
            "the backend '",
            handle.backend,
            "' is not currently supported",
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
    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var b_tensor = from_ndbuffer_row_major(b)

    matmul[use_tf32=use_tf32](
        ctx,
        handle,
        c_tensor,
        a_tensor,
        b_tensor,
        c_row_major=c_row_major,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        alpha=alpha,
        beta=beta,
    )


# ===----------------------------------------------------------------------===#
# CUBLAS
# ===----------------------------------------------------------------------===#


fn _cublas_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: cublasHandle_t,
    c: LayoutTensor[c_type, c_layout, _],
    a: LayoutTensor[a_type, a_layout, _],
    b: LayoutTensor[b_type, b_layout, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    comptime assert a_type == b_type and (
        a_type == DType.float32 or a_type.is_half_float()
    ), (
        "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
        " it if more types are needed."
    )

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1) if not transpose_a else a.dim(0)

    var c_dynamic_shape = IndexList[2](c.dim(0), c.dim(1))
    var a_dynamic_shape = IndexList[2](a.dim(0), a.dim(1))
    var b_dynamic_shape = IndexList[2](b.dim(0), b.dim(1))

    var compute_type: ComputeType

    comptime if a_type == DType.float16:
        compute_type = ComputeType.COMPUTE_32F
    elif a_type == DType.bfloat16:
        compute_type = ComputeType.COMPUTE_32F
    else:
        compute_type = (
            ComputeType.COMPUTE_32F_FAST_TF32 if use_tf32 else ComputeType.COMPUTE_32F
        )

    # When use_tf32 is True, CUBLAS will use TF32 to speedup the computation.
    # However, the result is not bit-wise identical to the result of FP32.
    comptime if use_tf32:
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
                Int32(N),
                Int32(M),
                Int32(K),
                LegacyUnsafePointer(to=alpha)
                .bitcast[NoneType]()
                .as_unsafe_pointer(),
                UnsafePointer(b.ptr.bitcast[NoneType]()),
                _convert_to_cublas_datatype[b_type](),
                Int32(K) if transpose_b else Int32(N),
                UnsafePointer(a.ptr.bitcast[NoneType]()),
                _convert_to_cublas_datatype[a_type](),
                Int32(M) if transpose_a else Int32(K),
                LegacyUnsafePointer(to=beta)
                .bitcast[NoneType]()
                .as_unsafe_pointer(),
                UnsafePointer(c.ptr.bitcast[NoneType]()),
                _convert_to_cublas_datatype[c_type](),
                Int32(N),
                compute_type,
                Algorithm.DEFAULT,
            ),
            msg=String(
                "failed to operate on cublas on the shape C=",
                c_dynamic_shape,
                "x",
                c_type,
                ", A=",
                a_dynamic_shape,
                "x",
                a_type,
                ", B=",
                b_dynamic_shape,
                "x",
                b_type,
            ),
        )
    # Default column-major.
    check_cublas_error(
        cublasGemmEx(
            handle,
            _convert_to_cublas_transpose(transpose_a),
            _convert_to_cublas_transpose(transpose_b),
            Int32(M),
            Int32(N),
            Int32(K),
            LegacyUnsafePointer(to=alpha)
            .bitcast[NoneType]()
            .as_unsafe_pointer(),
            UnsafePointer(a.ptr.bitcast[NoneType]()),
            _convert_to_cublas_datatype[a_type](),
            Int32(M),
            UnsafePointer(b.ptr.bitcast[NoneType]()),
            _convert_to_cublas_datatype[b_type](),
            Int32(N) if transpose_b else Int32(K),
            LegacyUnsafePointer(to=beta)
            .bitcast[NoneType]()
            .as_unsafe_pointer(),
            UnsafePointer(c.ptr.bitcast[NoneType]()),
            _convert_to_cublas_datatype[c_type](),
            Int32(M),
            compute_type,
            Algorithm.DEFAULT,
        ),
        msg=String(
            "failed to operate on cublas on the shape C=",
            c_dynamic_shape,
            "x",
            c_type,
            ", A=",
            a_dynamic_shape,
            "x",
            a_type,
            ", B=",
            b_dynamic_shape,
            "x",
            b_type,
        ),
    )


# ===----------------------------------------------------------------------===#
# ROCBLAS
# ===----------------------------------------------------------------------===#


fn _rocblas_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: _rocblas.Handle,
    c: LayoutTensor[c_type, c_layout, _],
    a: LayoutTensor[a_type, a_layout, _],
    b: LayoutTensor[b_type, b_layout, _],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    comptime assert a_type == b_type and (
        a_type == DType.float32 or a_type.is_half_float()
    ), (
        "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
        " it if more types are needed."
    )

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1) if not transpose_a else a.dim(0)

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
                Int32(N),
                Int32(M),
                Int32(K),
                LegacyUnsafePointer(to=alpha).bitcast[NoneType](),
                UnsafePointer(b.ptr.bitcast[NoneType]()),
                _rocblas.types.DataType(b_type),
                Int32(K) if transpose_b else Int32(N),
                UnsafePointer(a.ptr.bitcast[NoneType]()),
                _rocblas.types.DataType(a_type),
                Int32(K),
                LegacyUnsafePointer(to=beta).bitcast[NoneType](),
                UnsafePointer(c.ptr.bitcast[NoneType]()),
                _rocblas.types.DataType(c_type),
                Int32(N),
                UnsafePointer(c.ptr.bitcast[NoneType]()),
                _rocblas.types.DataType(c_type),
                Int32(N),
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
            Int32(M),
            Int32(N),
            Int32(K),
            LegacyUnsafePointer(to=alpha).bitcast[NoneType](),
            UnsafePointer(a.ptr.bitcast[NoneType]()),
            _rocblas.types.DataType(a_type),
            Int32(M),
            UnsafePointer(b.ptr.bitcast[NoneType]()),
            _rocblas.types.DataType(b_type),
            Int32(N) if transpose_b else Int32(K),
            LegacyUnsafePointer(to=beta).bitcast[NoneType](),
            UnsafePointer(c.ptr.bitcast[NoneType]()),
            _rocblas.types.DataType(c_type),
            Int32(M),
            UnsafePointer(c.ptr.bitcast[NoneType]()),
            _rocblas.types.DataType(c_type),
            Int32(M),
            compute_type,
            _rocblas.rocblas.types.Algorithm.STANDARD,
            0,
            0,
        )
    )


# ===----------------------------------------------------------------------===#
# CUBLASLT
# ===----------------------------------------------------------------------===#


fn _cublasLt_matmul[
    d_type: DType,
    a_type: DType,
    b_type: DType,
    d_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    scales_type: DType = DType.invalid,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    handle: UnsafePointer[Context],
    d: LayoutTensor[d_type, d_layout, _],
    a: LayoutTensor[a_type, a_layout, _],
    b: LayoutTensor[b_type, b_layout, _],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    var M = d.dim(0)
    var N = d.dim(1)
    var K = a.dim(1)

    comptime assert a_type in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
        DType.bfloat16,
        DType.float16,
        DType.uint8,
    ), (
        "Only E4M3, E5M2, bfloat16, float16, and E2M1x2 (UInt8) input data"
        " types are supported. Please extend it if you need more data"
        " types."
    )

    comptime assert a_type == b_type, "A and B must have the same type"

    comptime if a_type.is_float8():
        comptime assert not (a_type == b_type == DType.float8_e5m2), (
            "E5M2xE5m2 is not supported! Please refer to"
            " `https://docs.nvidia.com/cuda/cublas/#id105`"
        )

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
            LegacyUnsafePointer(to=compute_desc),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        ),
        msg="failed to create cublasLtMatmulDesc",
    )

    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA,
            LegacyUnsafePointer(to=transa).bitcast[NoneType](),
            size_of[cublasOperation_t](),
        ),
        msg="failed to set cublasLtMatmulDescAttribute for transa",
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB,
            LegacyUnsafePointer(to=transb).bitcast[NoneType](),
            size_of[cublasOperation_t](),
        ),
        msg="failed to set cublasLtMatmulDescAttribute for transb",
    )

    comptime if ctx.default_device_info.compute == B200.compute:
        if a_scales or b_scales:
            if not (a_scales and b_scales):
                raise Error("a_scales and b_scales must be provided together")
            a_scale_tensor = a_scales.value()
            b_scale_tensor = b_scales.value()

            comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE if scales_type == NVFP4_SF_DTYPE else MXFP8_SF_VECTOR_SIZE

            if scales_type not in (MXFP8_SF_DTYPE, NVFP4_SF_DTYPE):
                raise Error(
                    "Only float8_e8m0fnu(scaling type: MXFP8) and"
                    " float8_e4m3fn(scaling type: MXFP4) are supported for B200"
                )
            if not (
                a_type == b_type
                and (
                    (
                        a_type == DType.float8_e4m3fn
                        and scales_type == MXFP8_SF_DTYPE
                    )
                    or (a_type == DType.uint8 and scales_type == NVFP4_SF_DTYPE)
                )
            ):
                raise Error(
                    "Only E4M3 input with MXFP8 scales or E2M1x2(i.e,"
                    " UINT8) input with NVFP4 scales are supported for block"
                    " scaled matmul"
                )

            # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
            # We need to double the K dimension as we are allocating for uint8 input data type.
            # Remove this when GENAI-337 is fixed.
            if a_type == DType.uint8 and scales_type == NVFP4_SF_DTYPE:
                K = K * 2

            if not (
                (a_type == DType.uint8 and K % 32 == 0)
                or (a_type == DType.float8_e4m3fn and K % 16 == 0)
            ):
                raise Error(
                    "Due to TMA 16B alignment requirement, K must be divisible"
                    " by 16/32 for MXFP8/NVFP4 input data type, respectively"
                )

            if comptime (
                a_scales_layout.rank() != 5 or b_scales_layout.rank() != 5
            ):
                raise Error(
                    "Invalid A/B scales dimensions. Expected 5D tensors."
                )

            if (
                a_scale_tensor.dim(0) != ceildiv(M, SF_MN_GROUP_SIZE)
                or a_scale_tensor.dim(1)
                != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
                or b_scale_tensor.dim(0) != ceildiv(N, SF_MN_GROUP_SIZE)
                or b_scale_tensor.dim(1)
                != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
                or a_scale_tensor.dim(2)
                != b_scale_tensor.dim(2)
                != SF_ATOM_M[0]
                or a_scale_tensor.dim(3)
                != b_scale_tensor.dim(3)
                != SF_ATOM_M[1]
                or a_scale_tensor.dim(4) != b_scale_tensor.dim(4) != SF_ATOM_K
            ):
                raise Error("Invalid A/B scales dimensions.")

            var a_scale_mode: cublasLtMatmulMatrixScale_t
            var b_scale_mode: cublasLtMatmulMatrixScale_t

            a_scale_mode = b_scale_mode = (
                cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC16_UE4M3 if scales_type
                == NVFP4_SF_DTYPE else cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0
            )

            var a_scale_ptr = b_scale_tensor.ptr.bitcast[
                NoneType
            ]() if c_row_major else a_scale_tensor.ptr.bitcast[NoneType]()
            var b_scale_ptr = a_scale_tensor.ptr.bitcast[
                NoneType
            ]() if c_row_major else b_scale_tensor.ptr.bitcast[NoneType]()

            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    LegacyUnsafePointer(to=a_scale_mode).bitcast[NoneType](),
                    size_of[Int32](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix A"
                    " scale mode"
                ),
            )
            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    LegacyUnsafePointer(to=b_scale_mode).bitcast[NoneType](),
                    size_of[Int32](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix B"
                    " scale mode"
                ),
            )

            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    LegacyUnsafePointer(to=a_scale_ptr).bitcast[NoneType](),
                    size_of[UnsafePointer[NoneType]](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix A"
                    " scale factor"
                ),
            )
            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    LegacyUnsafePointer(to=b_scale_ptr).bitcast[NoneType](),
                    size_of[UnsafePointer[NoneType]](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix B"
                    " scale factor"
                ),
            )
    else:
        if a_scales or b_scales:
            raise Error("block scaling is only supported on B200 devices")

    # create matrix descriptors, we are good with the details here so no need to set any extra attributes
    # table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    var _adesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            LegacyUnsafePointer(to=_adesc),
            _convert_to_cublas_datatype[a_type](),
            UInt64(K),
            UInt64(N) if c_row_major else UInt64(M),
            Int64(K),
        ),
        msg="failed to create cublasLtMatrixLayout for adesc",
    )

    var _bdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            LegacyUnsafePointer(to=_bdesc),
            _convert_to_cublas_datatype[b_type](),
            UInt64(K),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(K),
        ),
        msg="failed to create cublasLtMatrixLayout for bdesc",
    )

    var _ddesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            LegacyUnsafePointer(to=_ddesc),
            _convert_to_cublas_datatype[d_type](),
            UInt64(N) if c_row_major else UInt64(M),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(N) if c_row_major else Int64(M),
        ),
        msg="failed to create cublasLtMatrixLayout for ddesc",
    )

    var _cdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            LegacyUnsafePointer(to=_cdesc),
            _convert_to_cublas_datatype[d_type](),
            UInt64(N) if c_row_major else UInt64(M),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(N) if c_row_major else Int64(M),
        ),
        msg="failed to create cublasLtMatrixLayout for cdesc",
    )

    var preference = cublasLtMatmulPreference_t()
    check_cublas_error(
        cublasLtMatmulPreferenceCreate(LegacyUnsafePointer(to=preference)),
        msg="failed to create cublasLtMatmulPreference",
    )

    var workspace_size = 32 * 1024 * 1024
    check_cublas_error(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            Preference.MAX_WORKSPACE_BYTES,
            LegacyUnsafePointer(to=workspace_size).bitcast[NoneType](),
            size_of[Int64](),
        ),
        msg=(
            "failed to set cublasLtMatmulPreferenceAttribute for"
            " max_workspace_bytes"
        ),
    )

    var heuristic_result = cublasLtMatmulHeuristicResult_t()
    var algorithm_count = 0
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
            LegacyUnsafePointer(to=heuristic_result),
            LegacyUnsafePointer(to=algorithm_count),
        ),
        msg="failed to get cublasLtMatmulAlgoGetHeuristic",
    )

    if algorithm_count == 0:
        raise Error("No algorithm was found!")

    var matmul_workspace = ctx.enqueue_create_buffer[DType.uint8](
        workspace_size
    )

    if c_row_major:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                LegacyUnsafePointer(to=alpha).bitcast[NoneType](),  # alpha
                UnsafePointer(b.ptr.bitcast[NoneType]()),  # _a
                _adesc,  # _adesc
                UnsafePointer(a.ptr.bitcast[NoneType]()),  # _b
                _bdesc,  # _bdesc
                LegacyUnsafePointer(to=beta).bitcast[NoneType](),  # beta
                OpaquePointer(),  # _c
                _cdesc,  # _cdesc
                UnsafePointer(d.ptr.bitcast[NoneType]()),  # _d
                _ddesc,  # _ddesc
                LegacyUnsafePointer(to=heuristic_result.algo),  # algo
                matmul_workspace.unsafe_ptr().bitcast[NoneType](),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream[],  # stream
            ),
            msg="failed to cublasLtMatmul for c_row_major=True",
        )
    else:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                LegacyUnsafePointer(to=alpha).bitcast[NoneType](),  # alpha
                UnsafePointer(a.ptr.bitcast[NoneType]()),  # _a
                _adesc,  # _adesc
                UnsafePointer(b.ptr.bitcast[NoneType]()),  # _b
                _bdesc,  # _bdesc
                LegacyUnsafePointer(to=beta).bitcast[NoneType](),  # beta
                OpaquePointer(),  # _c
                _cdesc,  # _cdesc
                UnsafePointer(d.ptr.bitcast[NoneType]()),  # _d
                _ddesc,  # _ddesc
                LegacyUnsafePointer(to=heuristic_result.algo),  # algo
                matmul_workspace.unsafe_ptr().bitcast[NoneType](),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream[],  # stream
            ),
            msg="failed to cublasLtMatmul for c_row_major=False",
        )

    check_cublas_error(
        cublasLtMatmulDescDestroy(compute_desc),
        msg="failed to destroy cublasLtMatmulDesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_adesc),
        msg="failed to destroy cublasLtMatrixLayout for adesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_bdesc),
        msg="failed to destroy cublasLtMatrixLayout for bdesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_cdesc),
        msg="failed to destroy cublasLtMatrixLayout for cdesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_ddesc),
        msg="failed to destroy cublasLtMatrixLayout for ddesc",
    )
    check_cublas_error(
        cublasLtMatmulPreferenceDestroy(preference),
        msg="failed to destroy cublasLtMatmulPreference",
    )

    _ = matmul_workspace^


# ===----------------------------------------------------------------------===#
# HIPBLASLT
# ===----------------------------------------------------------------------===#


fn _hipblasLt_matmul[
    d_type: DType,
    a_type: DType,
    b_type: DType,
    d_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
](
    ctx: DeviceContext,
    handle: hipblasLtHandle_t,
    d: LayoutTensor[d_type, d_layout, _],
    a: LayoutTensor[a_type, a_layout, _],
    b: LayoutTensor[b_type, b_layout, _],
    *,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    comptime assert a_type in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float8_e4m3fn,
        DType.float8_e5m2,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
    ), "Unsupported data type. Please extend it if you need more data types."

    comptime assert a_type == b_type, "A and B must have the same type"

    @always_inline
    @parameter
    fn create_matrix_layout[
        buf_type: DType,
        buf_layout: Layout,
    ](
        buf: LayoutTensor[buf_type, buf_layout, _]
    ) raises -> hipblasLtMatrixLayout_t:
        var _desc = hipblasLtMatrixLayout_t()
        _check_hipblas_error(
            hipblasLtMatrixLayoutCreate(
                LegacyUnsafePointer(to=_desc),
                _convert_to_hip_datatype[buf_type](),
                UInt64(buf.dim(1)),
                UInt64(buf.dim(0)),
                Int64(buf.dim(1)),
            )
        )
        return _desc

    @always_inline
    fn set_matrix_layout_batch_size(
        mat_layout: hipblasLtMatrixLayout_t,
        batch_size: Int,
        batch_stride: Int64,
    ) raises:
        _check_hipblas_error(
            hipblasLtMatrixLayoutSetAttribute(
                mat_layout,
                hipblasLtMatmulLayoutAttribute_t.BATCH_COUNT,
                LegacyUnsafePointer(to=batch_size).bitcast[NoneType](),
                size_of[Int](),
            )
        )
        _check_hipblas_error(
            hipblasLtMatrixLayoutSetAttribute(
                mat_layout,
                hipblasLtMatmulLayoutAttribute_t.STRIDED_BATCH_OFFSET,
                LegacyUnsafePointer(to=batch_stride).bitcast[NoneType](),
                size_of[Int64](),
            )
        )

    var _adata = UnsafePointer(a.ptr.bitcast[NoneType]())
    var _bdata = UnsafePointer(b.ptr.bitcast[NoneType]())
    var _ddata = UnsafePointer(d.ptr.bitcast[NoneType]())

    var _adesc = create_matrix_layout(a)
    var _bdesc = create_matrix_layout(b)
    var _ddesc = create_matrix_layout(d)

    # set batch size accordingly
    if batch_size > 1:
        set_matrix_layout_batch_size(
            _adesc, batch_size, Int64(a.dim(0) * a.dim(1))
        )
        set_matrix_layout_batch_size(
            _bdesc, batch_size, Int64(b.dim(0) * b.dim(1))
        )
        set_matrix_layout_batch_size(
            _ddesc, batch_size, Int64(d.dim(0) * d.dim(1))
        )

    var transa = (
        hipblasOperation_t.OP_T if transpose_a else hipblasOperation_t.OP_N
    )
    var transb = (
        hipblasOperation_t.OP_T if transpose_b else hipblasOperation_t.OP_N
    )

    # hipblasLt is by default column-major but we like to have the output in row-major
    # to compare with our results. Use `c_row_major` to determine the output layout.
    if c_row_major:
        swap(_adata, _bdata)
        swap(_adesc, _bdesc)
        swap(transa, transb)

    var operationDesc = hipblasLtMatmulDesc_t()
    _check_hipblas_error(
        hipblasLtMatmulDescCreate(
            LegacyUnsafePointer(to=operationDesc),
            hipblasComputeType_t.COMPUTE_32F,
            hipDataType_t.R_32F,
        )
    )

    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSA,
            LegacyUnsafePointer(to=transa).bitcast[NoneType](),
            size_of[hipblasOperation_t](),
        )
    )
    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSB,
            LegacyUnsafePointer(to=transb).bitcast[NoneType](),
            size_of[hipblasOperation_t](),
        )
    )

    var preference = hipblasLtMatmulPreference_t()
    _check_hipblas_error(
        hipblasLtMatmulPreferenceCreate(LegacyUnsafePointer(to=preference))
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
            LegacyUnsafePointer(to=heuristicResult),
            LegacyUnsafePointer(to=returnedResults),
        )
    )

    if returnedResults == 0:
        raise Error("No algorithm was found!")

    var workspace_size = heuristicResult.workspaceSize
    var workspace = ctx.enqueue_create_buffer[DType.uint8](workspace_size)

    _check_hipblas_error(
        hipblasLtMatmul(
            handle,
            operationDesc,
            LegacyUnsafePointer(to=alpha).bitcast[NoneType](),
            _adata,
            _adesc,
            _bdata,
            _bdesc,
            LegacyUnsafePointer(to=beta).bitcast[NoneType](),
            _ddata,
            _ddesc,
            _ddata,
            _ddesc,
            LegacyUnsafePointer(to=heuristicResult.algo),
            workspace.unsafe_ptr().bitcast[NoneType](),
            workspace_size,
            HIP(ctx.stream()),
        )
    )

    _check_hipblas_error(hipblasLtMatmulPreferenceDestroy(preference))
    _check_hipblas_error(hipblasLtMatmulDescDestroy(operationDesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_adesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_bdesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_ddesc))
