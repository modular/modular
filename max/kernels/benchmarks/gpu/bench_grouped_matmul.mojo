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

from os import abort
from math import ceildiv, align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    env_get_string,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    dynamic,
    init_vector_launch,
    static,
)
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_1d1d_nvfp4,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from layout._coord import Coord, Idx, RuntimeInt
from layout._layout import row_major
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout._tile_tensor import TileTensor
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    GMEMLayout1D,
)
from linalg.utils import elementwise_epilogue_type

from utils import Index, IndexList
from collections import Optional

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    set_scale_factor,
)


fn _get_run_name[
    in_type: DType,
    out_type: DType,
    *,
    use_vendor_blas: Bool,
    has_epilogue: Bool = False,
](num_active_experts: Int, total_num_tokens: Int, N: Int, K: Int) -> String:
    var vendor_str = "vendor_gmm" if use_vendor_blas else "gmm"
    var type_str = String("(", in_type, " -> ", out_type, ") : ")
    # num_active_experts
    var num_active_experts_str = String(num_active_experts)
    # total_num_tokens
    var total_num_tokens_str = String(total_num_tokens)
    # N
    var n_str = String(N)
    # K
    var k_str = String(K)
    # has_epilogue
    var has_epilogue_str = String(" with epilogue" if has_epilogue else "")

    return String(
        vendor_str,
        type_str,
        num_active_experts_str,
        " x ",
        total_num_tokens_str,
        " x ",
        n_str,
        " x ",
        k_str,
        has_epilogue_str,
    )


comptime epilogue_func_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
fn test_epilogue[
    dtype: DType
](m: Int, n: Int, val: Scalar[dtype]) -> Scalar[dtype]:
    return val + 4 * (Scalar[dtype]((m + n) % 21 - 10))


@always_inline
@parameter
fn add_two[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


fn bench_grouped_matmul[
    _in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
    scaling_kind_str: String = "1d2d",
    AB_swapped: Bool = False,
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_input: List[Int],
    init_type: InitializationType,
) raises:
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    comptime is_fp4e2m1 = _in_type == DType.float4_e2m1fn
    comptime in_type = DType.uint8 if is_fp4e2m1 else _in_type  # TODO: (KERN-2238): Replace with float4-e2m1fn
    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for num_tokens in num_tokens_by_expert:
        total_num_tokens += num_tokens
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, num_tokens)
    var M = total_num_tokens
    var total_flops = 2 * M * N * K

    # Define shapes and sizes
    # For fp4, data is stored as uint8 (2 fp4 values per byte), so K dimension is halved
    comptime packed_K = K // 2 if is_fp4e2m1 else K
    comptime static_a_shape = DimList(Dim(), packed_K)
    var a_size = total_num_tokens * packed_K
    comptime static_c_shape = DimList(Dim(), N)
    var c_size = total_num_tokens * N
    comptime static_b_shape = DimList(num_experts, N, packed_K)
    var b_size = num_experts * N * packed_K

    # Host allocations
    var a_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        num_active_experts
    )
    var expert_ids_host_ptr = UnsafePointer[Scalar[DType.int32]].alloc(
        num_active_experts
    )

    # Setup offsets and expert ids
    a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        var num_tokens = num_tokens_by_expert[i]
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(num_tokens)
        a_scale_dim0 += ceildiv(num_tokens, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids_input[i])

        @parameter
        if in_type == DType.float8_e4m3fn:
            comptime a_scale_alignment = 16 // size_of[DType.float32]()
            if num_tokens % a_scale_alignment != 0:
                abort(
                    "num_tokens=num_tokens_by_expert["
                    + String(i)
                    + "]="
                    + String(num_tokens)
                    + " must be divisible by a_scale_alignment="
                    + String(a_scale_alignment)
                )

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )

    var a_dev = NDBuffer[a_type, 2, _, static_a_shape](
        a_dev_buffer.unsafe_ptr(),
        DimList(total_num_tokens, packed_K),
    )
    var b_dev = NDBuffer[b_type, 3, _, static_b_shape](
        b_dev_buffer.unsafe_ptr(),
        static_b_shape,
    )
    var c_dev = NDBuffer[c_type, 2, _, static_c_shape](
        c_dev_buffer.unsafe_ptr(),
        DimList(total_num_tokens, N),
    )
    var a_offsets_dev = NDBuffer[DType.uint32, 1](
        a_offsets_dev_buffer.unsafe_ptr(),
        num_active_experts + 1,
    )
    var expert_ids_dev = NDBuffer[DType.int32, 1](
        expert_ids_dev_buffer.unsafe_ptr(),
        num_active_experts,
    )

    # Initialize data on the device
    init_vector_launch[a_type](a_dev_buffer, a_size, init_type, ctx)
    init_vector_launch[b_type](b_dev_buffer, b_size, init_type, ctx)

    # Move host-initialized data to device
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)

    @always_inline
    @__copy_capture(c_dev)
    @parameter
    fn epilogue_fn[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val

        @parameter
        for i in range(width):
            new_val[i] = test_epilogue(idx[0], idx[1] + i, val[i])

        c_dev.store[width=width, alignment=alignment](
            idx, new_val.cast[out_type]()
        )

    var a = from_ndbuffer_row_major(a_dev)
    var b = from_ndbuffer_row_major(b_dev)
    var c = from_ndbuffer_row_major(c_dev)
    var a_offsets = from_ndbuffer_row_major(a_offsets_dev)
    var expert_ids = from_ndbuffer_row_major(expert_ids_dev)

    @parameter
    if is_fp4e2m1:
        constrained[
            scaling_kind_str == "nvfp4",
            "Only support nvfp4 scaling kind for float4-e2m1fn",
        ]()

        var a_scale_offsets_dev_buffer = ctx.enqueue_create_buffer[
            DType.uint32
        ](num_active_experts)
        var a_scale_offsets_dev = NDBuffer[DType.uint32, 1](
            a_scale_offsets_dev_buffer.unsafe_ptr(), num_active_experts
        )
        ctx.enqueue_copy(a_scale_offsets_dev_buffer, a_scale_offsets_ptr)
        var a_scale_offsets = from_ndbuffer_row_major(a_scale_offsets_dev)

        # Calculate scales dimensions
        var a_scales_size = (
            a_scale_dim0
            * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )
        var b_scales_size = (
            num_experts
            * ceildiv(N, SF_MN_GROUP_SIZE)
            * ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
            * SF_ATOM_M[0]
            * SF_ATOM_M[1]
            * SF_ATOM_K
        )

        # Scales device allocations
        var a_scales_dev_buffer = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
            a_scales_size
        )
        var b_scales_dev_buffer = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](
            b_scales_size
        )

        init_vector_launch[NVFP4_SF_DTYPE](
            a_scales_dev_buffer,
            a_scales_size,
            init_type,
            ctx,
        )
        init_vector_launch[NVFP4_SF_DTYPE](
            b_scales_dev_buffer,
            b_scales_size,
            init_type,
            ctx,
        )

        # Build TileTensors for scale factors directly from device buffer
        # pointers + row_major layouts, bypassing NDBuffer entirely.
        # This avoids the compiler bug with ceildiv(...) in DimList.
        comptime k_groups = ceildiv(K, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
        var a_scales_tt = TileTensor(
            a_scales_dev_buffer.unsafe_ptr().bitcast[Scalar[NVFP4_SF_DTYPE]](),
            row_major(
                Coord(
                    RuntimeInt[DType.int64](Scalar[DType.int64](a_scale_dim0)),
                    Idx[k_groups](),
                    Idx[SF_ATOM_M[0]](),
                    Idx[SF_ATOM_M[1]](),
                    Idx[SF_ATOM_K](),
                )
            ),
        ).as_any_origin()
        var b_scales_tt = TileTensor(
            b_scales_dev_buffer.unsafe_ptr().bitcast[Scalar[NVFP4_SF_DTYPE]](),
            row_major(
                Coord(
                    Idx[num_experts](),
                    Idx[n_groups](),
                    Idx[k_groups](),
                    Idx[SF_ATOM_M[0]](),
                    Idx[SF_ATOM_M[1]](),
                    Idx[SF_ATOM_K](),
                )
            ),
        ).as_any_origin()

        var expert_scales_dev_buffer = ctx.enqueue_create_buffer[DType.float32](
            num_experts
        )
        var expert_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
            num_experts
        )
        for i in range(num_experts):
            expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(
                num_experts
            )
        ctx.enqueue_copy(expert_scales_dev_buffer, expert_scales_host_ptr)
        var expert_scales_tt = TileTensor(
            expert_scales_dev_buffer.unsafe_ptr().bitcast[
                Scalar[DType.float32]
            ](),
            row_major(
                Coord(
                    RuntimeInt[DType.int64](Scalar[DType.int64](num_experts)),
                )
            ),
        ).as_any_origin()

        @parameter
        @__copy_capture(
            a_dev,
            b_dev,
            c_dev,
            a_offsets_dev,
            expert_ids_dev,
            a_scale_offsets_dev,
            a_scales_tt,
            b_scales_tt,
            expert_scales_tt,
        )
        @always_inline
        fn bench_func_nvfp4(mut bench: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
                @parameter
                if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    comptime umma_shape = Index(128, 128, 32)
                    comptime transpose_b = True
                    comptime config = BlockScaledMatmulConfig[
                        a_type,
                        b_type,
                        c_type,
                        NVFP4_SF_DTYPE,
                        NVFP4_SF_DTYPE,
                        transpose_b,
                    ](
                        scaling_kind=UMMAKind.KIND_MXF4NVF4,
                        cluster_shape=Index(1, 1, 1),
                        mma_shape=umma_shape,
                        block_swizzle_size=8,
                        cta_group=1,
                        AB_swapped=AB_swapped,
                        k_group_size=1,
                        num_accum_pipeline_stages=2,
                    )
                    grouped_matmul_1d1d_nvfp4[
                        transpose_b=transpose_b,
                        config=config,
                    ](
                        TileTensor(c_dev),
                        TileTensor(a_dev),
                        TileTensor(a_offsets_dev),
                        TileTensor(a_scale_offsets_dev),
                        TileTensor(b_dev),
                        TileTensor(expert_ids_dev),
                        a_scales_tt,
                        b_scales_tt,
                        expert_scales_tt,
                        num_active_experts,
                        ctx,
                    )

            bench.iter_custom[kernel_launch](ctx)

        bench.bench_function[bench_func_nvfp4](
            BenchId(
                _get_run_name[
                    _in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

        _ = a_scales_dev_buffer^
        _ = b_scales_dev_buffer^
        _ = a_scale_offsets_dev_buffer^
        _ = expert_scales_dev_buffer^
        expert_scales_host_ptr.free()

    elif in_type == DType.float8_e4m3fn:
        constrained[
            scaling_kind_str == "1d2d",
            "Only support 1d2d scaling kind for float8_e4m3fn",
        ]()
        comptime BLOCK_SCALE_K = 128
        comptime static_a_scales_shape = DimList(K // BLOCK_SCALE_K, Dim())
        var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens
        comptime static_b_scales_shape = DimList(
            num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
        )
        var b_scales_size = (
            num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
        )

        # Scales device allocations
        var a_scales_dev_buffer = ctx.enqueue_create_buffer[DType.float32](
            a_scales_size
        )
        var b_scales_dev_buffer = ctx.enqueue_create_buffer[DType.float32](
            b_scales_size
        )

        var a_scales_dev = NDBuffer[DType.float32, 2, _, static_a_scales_shape](
            a_scales_dev_buffer.unsafe_ptr(),
            DimList(K // BLOCK_SCALE_K, total_num_tokens),
        )
        var b_scales_dev = NDBuffer[DType.float32, 3, _, static_b_scales_shape](
            b_scales_dev_buffer.unsafe_ptr(),
            static_b_scales_shape,
        )

        init_vector_launch[DType.float32](
            a_scales_dev_buffer,
            a_scales_size,
            init_type,
            ctx,
        )
        init_vector_launch[DType.float32](
            b_scales_dev_buffer,
            b_scales_size,
            init_type,
            ctx,
        )

        var a_scales = from_ndbuffer_row_major(a_scales_dev)
        var b_scales = from_ndbuffer_row_major(b_scales_dev)

        @parameter
        @__copy_capture(
            a_dev,
            b_dev,
            c_dev,
            a_offsets_dev,
            expert_ids_dev,
            a,
            b,
            c,
            a_scales,
            b_scales,
            a_offsets,
            expert_ids,
        )
        @always_inline
        fn bench_func_fp8_1d2d(mut bench: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
                @parameter
                if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    comptime umma_shape = Index(64, 64, 32)
                    comptime transpose_b = True
                    comptime config = MatmulConfig[
                        a_type, b_type, c_type, transpose_b
                    ](
                        cluster_shape=Index(1, 1, 1),
                        mma_shape=umma_shape,
                        cta_group=1,
                        AB_swapped=AB_swapped,
                        k_group_size=1,
                    )
                    grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
                        config=config,
                        elementwise_lambda_fn = Optional[
                            elementwise_epilogue_type
                        ](epilogue_fn) if has_epilogue else None,
                    ](
                        c,
                        a,
                        b,
                        a_scales,
                        b_scales,
                        a_offsets,
                        expert_ids,
                        max_num_tokens_by_expert,
                        num_active_experts,
                        ctx,
                    )

            bench.iter_custom[kernel_launch](ctx)

        bench.bench_function[bench_func_fp8_1d2d](
            BenchId(
                _get_run_name[
                    in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            # TODO: Pick relevant benchmetric
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

        _ = a_scales_dev_buffer^
        _ = b_scales_dev_buffer^
    else:

        @parameter
        @__copy_capture(
            a_dev,
            b_dev,
            c_dev,
            a_offsets_dev,
            expert_ids_dev,
            a,
            b,
            c,
            a_offsets,
            expert_ids,
        )
        @always_inline
        fn bench_func(mut bench: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
                @parameter
                if use_vendor_blas:
                    # TODO: Implement vendor grouped matmul
                    pass

                else:
                    grouped_matmul[
                        elementwise_lambda_fn = Optional[
                            elementwise_epilogue_type
                        ](epilogue_fn) if has_epilogue else None,
                    ](
                        c_dev,
                        a_dev,
                        b_dev,
                        a_offsets_dev,
                        expert_ids_dev,
                        max_num_tokens_by_expert,
                        num_active_experts,
                        ctx,
                    )

            bench.iter_custom[kernel_launch](ctx)

        bench.bench_function[bench_func](
            BenchId(
                _get_run_name[
                    in_type,
                    out_type,
                    use_vendor_blas=use_vendor_blas,
                    has_epilogue=has_epilogue,
                ](
                    num_active_experts,
                    total_num_tokens,
                    N,
                    K,
                )
            ),
            # TODO: Pick relevant benchmetric
            [
                ThroughputMeasure(
                    BenchMetric.flops,
                    total_flops,
                )
            ],
        )

    # Cleanup host pointers
    a_offsets_host_ptr.free()
    a_scale_offsets_ptr.free()
    expert_ids_host_ptr.free()

    # Consume device buffers
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


fn create_grouped_matmul_bench[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
    scaling_kind_str: String = "1d2d",
    AB_swapped: Bool = False,
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    init_type: InitializationType,
) raises:
    bench_grouped_matmul[
        in_type,
        out_type,
        num_experts,
        expert_shape,
        use_vendor_blas=use_vendor_blas,
        has_epilogue=has_epilogue,
        scaling_kind_str=scaling_kind_str,
        AB_swapped=AB_swapped,
    ](
        ctx,
        bench,
        num_active_experts,
        num_tokens_by_expert,
        expert_ids,
        init_type,
    )


fn string_to_list(string: String) raises -> List[Int]:
    var list = List[Int]()
    for i in string.split(","):
        try:
            list.append(Int(i))
        except:
            continue
    return list^


def main():
    comptime in_type = env_get_dtype["in_type", DType.bfloat16]()
    comptime out_type = env_get_dtype["out_type", DType.bfloat16]()
    comptime scaling_kind_str = env_get_string["scaling_kind", "1d2d"]()

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var num_tokens_by_expert_string = String(
        arg_parse("num_tokens_by_expert", "256")
    )
    var expert_ids_string = String(arg_parse("expert_ids", "0"))

    var num_tokens_by_expert = string_to_list(num_tokens_by_expert_string)
    var expert_ids = string_to_list(expert_ids_string)

    comptime N = env_get_int["N", 256]()
    comptime K = env_get_int["K", 256]()
    comptime num_experts = env_get_int["num_experts", 1]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime use_vendor_blas = env_get_bool["use_vendor_blas", False]()
    comptime has_epilogue = env_get_bool["has_epilogue", False]()
    comptime AB_swapped = env_get_bool["AB_swapped", False]()

    var b = Bench()
    comptime expert_shape = IndexList[2](N, K)

    with DeviceContext() as ctx:
        create_grouped_matmul_bench[
            in_type,
            out_type,
            num_experts,
            expert_shape,
            use_vendor_blas=use_vendor_blas,
            has_epilogue=has_epilogue,
            scaling_kind_str=scaling_kind_str,
            AB_swapped=AB_swapped,
        ](
            ctx,
            b,
            num_active_experts,
            num_tokens_by_expert,
            expert_ids,
            init_type,
        )

    b.dump_report()
