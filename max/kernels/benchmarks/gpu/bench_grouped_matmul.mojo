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

from math import align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from internal_utils import DeviceNDBuffer, HostNDBuffer, arg_parse
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    init_vector_launch,
    initialize,
    random,
    static,
)
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig

from utils import Index, IndexList
from collections import OptionalReg


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


alias epilogue_func_type = fn[dtype: DType, width: Int, *, alignment: Int = 1] (
    SIMD[dtype, width]
) capturing -> SIMD[dtype, width]


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
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    init_type: InitializationType,
) raises:
    alias N = expert_shape[0]
    alias K = expert_shape[1]

    alias a_type = in_type
    alias b_type = in_type
    alias c_type = out_type

    # Total and max number of tokens
    total_num_tokens = 0
    max_num_tokens_by_expert = 0
    for num_tokens in num_tokens_by_expert:
        total_num_tokens += num_tokens
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, num_tokens)

    # Create host A C buffers
    alias static_a_shape = DimList(Dim(), K)
    var dynamic_a_shape = DimList(total_num_tokens, K)
    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_c_shape = DimList(total_num_tokens, N)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var a_offsets_host = HostNDBuffer[DType.uint32, 1, DimList(Dim())](
        num_active_experts + 1
    )

    # Create host B buffers
    alias static_b_shape = DimList(num_experts, N, K)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](static_b_shape)
    var expert_ids_host = HostNDBuffer[DType.int32, 1](num_active_experts)

    # Setup offsets and expert ids
    a_offsets_host.tensor[0] = 0
    for i in range(num_active_experts):
        a_offsets_host.tensor[i + 1] = (
            a_offsets_host.tensor[i] + num_tokens_by_expert[i]
        )
        expert_ids_host.tensor[i] = expert_ids[i]

    # Create device buffers
    var a_dev = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var c_dev = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var b_dev = DeviceNDBuffer[b_type, 3, static_b_shape](
        static_b_shape, ctx=ctx
    )
    var a_offsets_dev = DeviceNDBuffer[DType.uint32, 1](
        num_active_experts + 1, ctx=ctx
    )
    var expert_ids_dev = DeviceNDBuffer[DType.int32, 1](
        num_active_experts, ctx=ctx
    )

    # Initialize data on the device
    init_vector_launch[a_type](
        a_dev.buffer, a_host.tensor.num_elements(), init_type, ctx
    )
    init_vector_launch[b_type](
        b_dev.buffer, b_host.tensor.num_elements(), init_type, ctx
    )

    # Move host-initialized data to device
    ctx.enqueue_copy(a_offsets_dev.buffer, a_offsets_host.tensor.data)
    ctx.enqueue_copy(expert_ids_dev.buffer, expert_ids_host.tensor.data)

    var c_dev_ndbuffer = c_dev.tensor

    @always_inline
    @__copy_capture(c_dev_ndbuffer)
    @parameter
    fn epilogue_fn[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val

        @parameter
        for i in range(width):
            new_val[i] = test_epilogue(idx[0], idx[1] + i, val[i])

        c_dev_ndbuffer.store[width=width, alignment=alignment](
            idx, new_val.cast[out_type]()
        )

    @parameter
    @__copy_capture(a_dev, b_dev, c_dev, a_offsets_dev, expert_ids_dev)
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
                    elementwise_lambda_fn = OptionalReg[
                        elementwise_epilogue_type
                    ](epilogue_fn) if has_epilogue else None,
                ](
                    c_dev.tensor,
                    a_dev.tensor,
                    b_dev.tensor,
                    a_offsets_dev.tensor,
                    expert_ids_dev.tensor,
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
        ThroughputMeasure(
            BenchMetric.flops,
            2 * total_num_tokens * N * K,
        ),
    )

    # Retain our buffers till the end.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^

    _ = a_host^
    _ = b_host^
    _ = c_host^


fn create_grouped_matmul_bench[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    /,
    *,
    use_vendor_blas: Bool = False,
    has_epilogue: Bool = False,
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
    alias in_type = env_get_dtype["in_type", DType.bfloat16]()
    alias out_type = env_get_dtype["out_type", DType.bfloat16]()

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var num_tokens_by_expert_string = String(
        arg_parse("num_tokens_by_expert", "256")
    )
    var expert_ids_string = String(arg_parse("expert_ids", "0"))

    var num_tokens_by_expert = string_to_list(num_tokens_by_expert_string)
    var expert_ids = string_to_list(expert_ids_string)

    alias N = env_get_int["N", 256]()
    alias K = env_get_int["K", 256]()
    alias num_experts = env_get_int["num_experts", 1]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    alias use_vendor_blas = env_get_bool["use_vendor_blas", False]()
    alias has_epilogue = env_get_bool["has_epilogue", False]()

    var b = Bench()
    alias expert_shape = IndexList[2](N, K)

    with DeviceContext() as ctx:
        create_grouped_matmul_bench[
            in_type,
            out_type,
            num_experts,
            expert_shape,
            use_vendor_blas=use_vendor_blas,
            has_epilogue=has_epilogue,
        ](
            ctx,
            b,
            num_active_experts,
            num_tokens_by_expert,
            expert_ids,
            init_type,
        )

    b.dump_report()
