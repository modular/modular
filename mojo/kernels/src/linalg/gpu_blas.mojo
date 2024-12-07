# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from sys import sizeof

from buffer import DimList, NDBuffer
from gpu.cublas.cublas import (
    Algorithm,
    ComputeType,
    _convert_to_cublas_datatype,
    _convert_to_cublas_transpose,
    check_cublas_error,
    cublasContext,
    cublasGemmEx,
    cublasOperation_t,
)
from gpu.cublas.cublaslt import (
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
from gpu.cublas.dtype import DataType
from gpu.cublas.result import Result
from gpu.host import DeviceContext
from gpu.host.nvidia_cuda import CUDA
from layout import Layout
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from memory import UnsafePointer


fn vendor_matmul[
    use_tf32: Bool = False,
](
    handle: UnsafePointer[cublasContext],
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
) -> Result:
    return _cublas_matmul[use_tf32=use_tf32](
        handle, c, a, b, c_row_major, transpose_a, transpose_b
    )


fn _cublas_matmul[
    use_tf32: Bool = False,
](
    handle: UnsafePointer[cublasContext],
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
) -> Result:
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

    var alpha = Scalar[DType.float32](1.0)
    var beta = Scalar[DType.float32](0.0)

    var compute_type: ComputeType

    @parameter
    if a.type == DType.float16:
        compute_type = ComputeType.COMPUTE_32F
    elif a.type == DType.bfloat16:
        compute_type = ComputeType.COMPUTE_32F
    else:
        compute_type = (
            ComputeType.COMPUTE_32F_FAST_TF32 if use_tf32 else ComputeType.COMPUTE_32F
        )

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

    var result: Result
    if c_row_major:
        result = cublasGemmEx(
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
        )
    # Default column-major.
    else:
        result = cublasGemmEx(
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
        )
    return result


fn _cublasLt_matmul(
    ctx: DeviceContext,
    d: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    c_row_major: Bool = True,
) raises -> Result:
    alias a_type = a.type
    alias b_type = b.type
    alias d_type = d.type
    var M = d.dim[0]()
    var N = d.dim[1]()
    var K = a.dim[1]()

    constrained[
        (
            a_type in [DType.float8e4m3, DType.float8e5m2]
            and b_type in [DType.float8e4m3, DType.float8e5m2]
        ),
        (
            "Only E4M3 and E5M2 input data types are supported. Please extend"
            " it if you need more data types."
        ),
    ]()

    if a_type is DType.float8e5m2 and b_type is DType.float8e4m3:
        raise Error("E5M2xE4M3 is not supported!")

    var lt_handle = UnsafePointer[Context]()
    check_cublas_error(cublasLtCreate(UnsafePointer.address_of(lt_handle)))

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

    # set the transforms for A and B
    var transa = cublasOperation_t.CUBLAS_OP_T
    var transb = cublasOperation_t.CUBLAS_OP_N

    var alpha = Scalar[DType.float32](1.0)
    var beta = Scalar[DType.float32](0.0)

    # create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    var operationDesc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer.address_of(operationDesc),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        )
    )

    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            operationDesc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA,
            UnsafePointer.address_of(transa).bitcast[NoneType](),
            sizeof[cublasOperation_t](),
        )
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            operationDesc,
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

    var workspaceSize = 0
    # var workspaceSize = 32 * 1024 * 1024
    # check_cublas_error(
    #     cublasLtMatmulPreferenceSetAttribute(
    #         preference,
    #         Preference.MAX_WORKSPACE_BYTES,
    #         UnsafePointer.address_of(workspaceSize).bitcast[NoneType](),
    #         sizeof[Int]()
    #     )
    # )

    var heuristicResult = cublasLtMatmulHeuristicResult_t()
    var returnedResults = 0
    check_cublas_error(
        cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            operationDesc,
            _adesc,
            _bdesc,
            _cdesc,
            _ddesc,
            preference,
            1,
            UnsafePointer.address_of(heuristicResult),
            UnsafePointer.address_of(returnedResults),
        )
    )

    if returnedResults == 0:
        raise Error("No algorithm was found!")

    var cuda_stream = CUDA(ctx.stream())

    var result: Result
    if c_row_major:
        result = cublasLtMatmul(
            lt_handle,  # light_handle
            operationDesc,  # compute_desc
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
            UnsafePointer.address_of(heuristicResult.algo),  # algo
            UnsafePointer[NoneType](),  # workspace
            workspaceSize,  # workspace_size_in_bytes
            cuda_stream[],  # stream
        )
    else:
        result = cublasLtMatmul(
            lt_handle,  # light_handle
            operationDesc,  # compute_desc
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
            UnsafePointer.address_of(heuristicResult.algo),  # algo
            UnsafePointer[NoneType](),  # workspace
            workspaceSize,  # workspace_size_in_bytes
            cuda_stream[],  # stream
        )

    ctx.synchronize()

    check_cublas_error(cublasLtMatmulDescDestroy(operationDesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_adesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_bdesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_cdesc))
    check_cublas_error(cublasLtMatrixLayoutDestroy(_ddesc))
    check_cublas_error(cublasLtMatmulPreferenceDestroy(preference))
    check_cublas_error(cublasLtDestroy(lt_handle))

    return result
