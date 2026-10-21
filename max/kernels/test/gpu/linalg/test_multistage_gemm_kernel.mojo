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

from max.gpu.host import DeviceContext, FuncAttribute
from std.math import ceildiv
from std.random import rand

from internal_utils import assert_almost_equal
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.tensor_engine import DefaultEngine, DevicePointerEngine
from linalg.matmul.gpu import matmul_kernel_naive, multistage_gemm
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulConfig, MatmulKernels


def multistage_gemm_simple[
    M: Int,
    N: Int,
    K: Int,
    a_type: DType = .bfloat16,
    b_type: DType = .bfloat16,
    c_type: DType = .bfloat16,
    transpose_b: Bool = False,
    # Defaulted because this helper only compiles the kernel: there is no tile
    # argument the engine could be inferred from.
    engine: TensorEngine = DefaultEngine[element_width=1],
](ctx: DeviceContext,) raises:
    comptime kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()
    comptime config = kernels.ampere_128x128_4

    comptime a_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    # Select the b dims first (an `Int` ternary unifies cleanly); a ternary
    # over the two `RowMajorLayout[...]` types directly does not, since the
    # transposed/non-transposed layouts are distinct parameterized types.
    comptime b_dim0 = N if transpose_b else K
    comptime b_dim1 = K if transpose_b else N
    comptime b_tt_layout = RowMajorLayout[
        ComptimeInt[b_dim0], ComptimeInt[b_dim1]
    ]
    comptime c_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]

    # Dispatch w/o split K
    comptime gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        c_tt_layout,
        a_type,
        a_tt_layout,
        b_type,
        b_tt_layout,
        transpose_b,
        c_linear_idx_type=.int64,
        a_linear_idx_type=.int64,
        b_linear_idx_type=.int64,
        c_engine=engine,
        a_engine=engine,
        b_engine=engine,
        config=config,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )


def multistage_gemm_run[
    M: Int,
    N: Int,
    K: Int,
    *,
    in_type: DType = .float32,
    out_type: DType = .float32,
    use_epilogue: Bool = False,
    num_k_partitions: Int = 1,
    runtime_k: Bool = False,
](ctx: DeviceContext) raises:
    """Runs the multistage GEMM and checks it against the naive kernel.

    An fp32 C with odd N has 4-byte-misaligned rows, which the kernel's usual
    8-byte vector store cannot write. The vendor BLAS fallback reaches this
    kernel with exactly that shape for GPT-2's lm_head (vocab 50257). With
    `num_k_partitions > 1` the split-K path stages the partials in an fp32
    workspace, so a half-precision C exercises the fp32-output kernel. With
    `runtime_k` the K extent of A and B is dynamic, as the split-K partitions
    are, which makes both operand copies masked.
    """
    comptime transpose_b = True

    print(
        "M:",
        M,
        "N:",
        N,
        "K:",
        K,
        in_type,
        "->",
        out_type,
        "use_epilogue:",
        use_epilogue,
        "num_k_partitions:",
        num_k_partitions,
        "runtime_k:",
        runtime_k,
    )

    var a_shape = row_major(Idx[M], Idx[K])
    var b_shape = row_major(Idx[N], Idx[K])
    var a_runtime_shape = row_major(Idx[M], K)
    var b_runtime_shape = row_major(Idx[N], K)
    var c_shape = row_major(Idx[M], Idx[N])

    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[in_type](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_type](M * N)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[out_type](M * N)

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[in_type](M * K)
    var b_device = ctx.enqueue_create_buffer[in_type](N * K)
    var c_device = ctx.enqueue_create_buffer[out_type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[out_type](M * N)

    var a_tensor = TileTensor(a_device, a_shape)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    rand(a_host.as_span())
    rand(b_host.as_span())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    def compute_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        return val

    # The config the vendor BLAS fallback hardcodes (KERN-1812).
    comptime config = MatmulKernels[
        in_type, in_type, out_type, transpose_b
    ]().ampere_256x64_4

    var a_dyn = TileTensor(a_device, a_runtime_shape)
    var b_dyn = TileTensor(b_device, b_runtime_shape)

    comptime if runtime_k:
        comptime assert num_k_partitions == 1 and not use_epilogue
        multistage_gemm[transpose_b=transpose_b, config=config](
            c_tensor, a_dyn, b_dyn, ctx
        )
    elif num_k_partitions > 1:
        comptime split_config = MatmulConfig[
            in_type, in_type, out_type, transpose_b
        ](
            block_tile_shape=config.block_tile_shape,
            warp_tile_shape=config.warp_tile_shape,
            num_pipeline_stages=config.num_pipeline_stages,
            num_k_partitions=num_k_partitions,
        )
        comptime if use_epilogue:
            multistage_gemm[transpose_b=transpose_b, config=config](
                c_tensor, a_tensor, b_tensor, split_config, compute_fn, ctx
            )
        else:
            multistage_gemm[transpose_b=transpose_b, config=config](
                c_tensor, a_tensor, b_tensor, split_config, ctx
            )
    elif use_epilogue:
        multistage_gemm[transpose_b=transpose_b, config=config](
            c_tensor, a_tensor, b_tensor, compute_fn, ctx
        )
    else:
        multistage_gemm[transpose_b=transpose_b, config=config](
            c_tensor, a_tensor, b_tensor, ctx
        )

    # cuBLAS rejects these shapes, so the reference is the naive GPU kernel,
    # the same one the vendor BLAS fallback uses for small K.
    comptime BLOCK_DIM = 16
    comptime naive_kernel = matmul_kernel_naive[
        out_type,
        in_type,
        in_type,
        type_of(c_ref_tensor).LayoutType,
        type_of(a_tensor).LayoutType,
        type_of(b_tensor).LayoutType,
        BLOCK_DIM,
        transpose_b,
        c_engine=type_of(c_ref_tensor).Engine,
        a_engine=type_of(a_tensor).Engine,
        b_engine=type_of(b_tensor).Engine,
    ]
    ctx.enqueue_function[naive_kernel](
        c_ref_tensor,
        a_tensor,
        b_tensor,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime if out_type.is_float8():
        # The host has no fp8 arithmetic, so compare in fp32. e4m3 keeps 3
        # mantissa bits: a one-ulp rounding difference from the naive
        # kernel's summation order is a 12.5% relative error.
        var c_f32 = List[Float32](capacity=M * N)
        var c_ref_f32 = List[Float32](capacity=M * N)
        for i in range(M * N):
            c_f32.append(c_host_ptr[i].cast[.float32]())
            c_ref_f32.append(c_host_ref_ptr[i].cast[.float32]())
        assert_almost_equal(Span(c_f32), Span(c_ref_f32), atol=1e-2, rtol=0.125)
    else:
        assert_almost_equal(
            c_host._storage,
            c_host_ref._storage,
            c_host.num_elements(),
            atol=1e-4 if out_type == .float32 else 1e-2,
            rtol=1e-2,
        )

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    with DeviceContext() as ctx:
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            .bfloat16,
            .bfloat16,
            .bfloat16,
            False,
        ](ctx)
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            .bfloat16,
            .bfloat16,
            .bfloat16,
            False,
        ](ctx)

        multistage_gemm_simple[
            550, 2048, 8, .float32, .float32, .float32, False
        ](ctx)

        # The same kernel instantiated with device-pointer tiles.
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            .bfloat16,
            .bfloat16,
            .bfloat16,
            False,
            engine=DevicePointerEngine[element_width=1],
        ](ctx)

        # fp32 C shapes the vectorized store cannot take: an odd N misaligns
        # every other row, and an N that is not a multiple of the block tile
        # leaves the trailing block running past the last column. NVIDIA-only,
        # since only that path vectorizes C along N.
        comptime if ctx.target.is_nvidia_gpu():
            # Odd N.
            multistage_gemm_run[M=20, N=257, K=768](ctx)
            multistage_gemm_run[M=20, N=257, K=768, use_epilogue=True](ctx)
            multistage_gemm_run[M=20, N=17, K=768](ctx)
            # GPT-2's lm_head, the shape that first hit the fault.
            multistage_gemm_run[M=20, N=50257, K=768](ctx)
            multistage_gemm_run[M=20, N=50257, K=768, use_epilogue=True](ctx)
            # Even N, partial trailing block.
            multistage_gemm_run[M=20, N=48, K=768](ctx)
            multistage_gemm_run[M=20, N=306, K=768](ctx)
            multistage_gemm_run[M=20, N=50256, K=768](ctx)
            multistage_gemm_run[M=20, N=50226, K=768, use_epilogue=True](ctx)
            # Exact block multiples: the vectorized store path.
            multistage_gemm_run[M=20, N=256, K=768](ctx)
            multistage_gemm_run[M=20, N=512, K=768, use_epilogue=True](ctx)
            # Half-precision inputs: the staged bf16 store, the fp32 store of
            # bf16 accumulators, dynamic-K operands, and split-K through the
            # fp32 workspace.
            multistage_gemm_run[
                M=130, N=512, K=512, in_type=.bfloat16, out_type=.bfloat16
            ](ctx)
            multistage_gemm_run[
                M=130, N=512, K=512, in_type=.bfloat16, out_type=.float32
            ](ctx)
            multistage_gemm_run[M=130, N=512, K=512, runtime_k=True](ctx)
            multistage_gemm_run[
                M=130,
                N=512,
                K=512,
                in_type=.bfloat16,
                out_type=.bfloat16,
                runtime_k=True,
            ](ctx)
            multistage_gemm_run[
                M=130,
                N=512,
                K=512,
                in_type=.bfloat16,
                out_type=.float32,
                runtime_k=True,
            ](ctx)
            multistage_gemm_run[
                M=130,
                N=512,
                K=512,
                in_type=.bfloat16,
                out_type=.bfloat16,
                num_k_partitions=2,
            ](ctx)
            multistage_gemm_run[M=130, N=512, K=512, num_k_partitions=2](ctx)
            # A C dtype other than the fp32 accumulator and not half precision
            # takes the fp32 store branch, which must cast rather than copy.
            multistage_gemm_run[
                M=130, N=512, K=512, in_type=.bfloat16, out_type=.float8_e4m3fn
            ](ctx)
            multistage_gemm_run[
                M=130,
                N=512,
                K=512,
                in_type=.bfloat16,
                out_type=.float8_e4m3fn,
                num_k_partitions=2,
            ](ctx)
