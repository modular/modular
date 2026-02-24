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

import pytest
import torch
import triton
import triton.language as tl
from max.driver import Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import (
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from max.nn.kernels import (
    dynamic_scaled_matmul,
    quantize_dynamic_scaled_float8,
)
from test_common.graph_utils import is_b100_b200, is_h100_h200

# TODO: confirm this is a reasonable rtol for fp8
ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-2


# The following triton kernels are adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    s_ptr: torch.Tensor,
    scale_ub: float,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    max_val = tl.max(tl.abs(x), axis=0)
    s = tl.minimum(max_val, scale_ub) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_ub: float = 1200.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        *x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32
    )
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](
        x, y, s, scale_ub, BLOCK_SIZE=tl.constexpr(block_size)
    )
    return y, s


@triton.jit
def fp8_gemm_kernel(
    a_ptr: torch.Tensor,
    b_ptr: torch.Tensor,
    c_ptr: torch.Tensor,
    a_s_ptr: torch.Tensor,
    b_s_ptr: torch.Tensor,
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
) -> None:
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * (K // TILE_SIZE_K)
    b_s_ptrs = b_s_ptr + (offs_n // TILE_SIZE_N) * (K // TILE_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0
        )
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

        if (i + 1) % (TILE_SIZE_K // BLOCK_SIZE_K) == 0:
            a_s_ptrs += 1
            b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(  # noqa: ANN201
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), (
        "Input tensors must be contiguous"
    )
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    tile_k = K // a_s.size(-1)
    tile_n = N // b_s.size(0)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_kernel[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        M,
        tl.constexpr(N),
        tl.constexpr(K),
        BLOCK_SIZE_M=tl.constexpr(64),
        BLOCK_SIZE_N=tl.constexpr(128),
        BLOCK_SIZE_K=tl.constexpr(128),
        TILE_SIZE_K=tl.constexpr(tile_k),
        TILE_SIZE_N=tl.constexpr(tile_n),
    )
    return c


def create_triton_fp8_result(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    weight_scales: torch.Tensor,
    a_tile_size: tuple[int, int],
    b_tile_size: tuple[int, int],
) -> torch.Tensor:
    input_quantized, input_scales = act_quant(
        input_tensor, block_size=a_tile_size[1]
    )
    result = fp8_gemm(
        input_quantized, input_scales, weight_tensor, weight_scales
    )
    return result


def create_max_fp8_result(
    session: InferenceSession,
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    weight_scales: torch.Tensor,
    a_tile_size: tuple[int, int],
    b_tile_size: tuple[int, int],
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
) -> torch.Tensor:
    K = input_tensor.shape[-1]
    N = weight_tensor.shape[0]

    def _build_graph():  # noqa: ANN202
        with Graph(
            "fp8_matmul",
            input_types=(
                TensorType(DType.bfloat16, ["M", K], device=DeviceRef.GPU()),
                TensorType(DType.float8_e4m3fn, [N, K], device=DeviceRef.GPU()),
                TensorType(
                    DType.float32,
                    [N // b_tile_size[0], K // b_tile_size[1]],
                    device=DeviceRef.GPU(),
                ),
            ),
        ) as graph:
            (input, weight, weight_scales) = [v.tensor for v in graph.inputs]

            quantized_input, input_scales = quantize_dynamic_scaled_float8(
                input,
                input_scale_spec,
                weight_scale_spec,
                group_size_or_per_token=a_tile_size[1]
                if a_tile_size[1] != K
                else -1,
                scales_type=DType.float32,
            )

            result = dynamic_scaled_matmul(
                quantized_input,
                weight,
                input_scales,
                weight_scales,
                input_scale_spec,
                weight_scale_spec,
            )
            graph.output(result)
        return graph

    graph = _build_graph()
    compiled = session.load(graph)

    # we need to view the torch fp8 tensor as uint8, as torch.Tensor doesn't support
    # to_dlpack() for fp8 dtype.
    _weight_tensor = weight_tensor.view(torch.uint8)
    _max_weight_tensor = Buffer.from_dlpack(_weight_tensor).view(
        DType.float8_e4m3fn, weight_tensor.shape
    )

    max_result = compiled.execute(
        input_tensor, _max_weight_tensor, weight_scales
    )[0]
    return torch.from_dlpack(max_result)


@pytest.mark.skipif(
    not (is_h100_h200() or is_b100_b200()),
    reason="float8 requires H100 or H200",
)
@pytest.mark.parametrize(
    "M, N, K, a_tile_size, b_tile_size, input_scale_granularity, weight_scale_granularity",
    [
        # per-token quantization for the input tensor, and per-column quantization
        # for the weight tensor
        (
            1,
            1024,
            2048,
            (1, 2048),
            (1, 2048),
            Float8ScaleGranularity.COLWISE,
            Float8ScaleGranularity.ROWWISE,
        ),
        (
            32,
            1024,
            2048,
            (1, 2048),
            (1, 2048),
            Float8ScaleGranularity.COLWISE,
            Float8ScaleGranularity.ROWWISE,
        ),
        (
            81,
            1024,
            2048,
            (1, 2048),
            (1, 2048),
            Float8ScaleGranularity.COLWISE,
            Float8ScaleGranularity.ROWWISE,
        ),
        # per-token quantization scaling for the input tensor, and 2D quantization
        # scaling for the weight tensor
        (
            1,
            1024,
            2048,
            (1, 128),
            (128, 128),
            Float8ScaleGranularity.BLOCK,
            Float8ScaleGranularity.BLOCK,
        ),
        (
            32,
            1024,
            2048,
            (1, 128),
            (128, 128),
            Float8ScaleGranularity.BLOCK,
            Float8ScaleGranularity.BLOCK,
        ),
        (
            81,
            1024,
            2048,
            (1, 128),
            (128, 128),
            Float8ScaleGranularity.BLOCK,
            Float8ScaleGranularity.BLOCK,
        ),
    ],
)
def test_linear_gpu(
    gpu_session: InferenceSession,
    M: int,
    N: int,
    K: int,
    a_tile_size: tuple[int, int],
    b_tile_size: tuple[int, int],
    input_scale_granularity: Float8ScaleGranularity,
    weight_scale_granularity: Float8ScaleGranularity,
) -> None:
    assert a_tile_size[0] == 1, "a_tile_size[0] must be 1"
    assert a_tile_size[1] == b_tile_size[1], (
        "a and b must have the same tile size in the K dimension"
    )
    torch.manual_seed(42)

    input_tensor = torch.randn(
        size=(M, K),
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    # This creates FP8_E4M3FN values between -448 and 448
    weight_tensor = (
        torch.randn(
            size=(N, K),
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        ).clamp(-448.0, 448.0)
    ).to(torch.float8_e4m3fn)

    weight_scales = (
        torch.rand(
            size=(N // b_tile_size[0], K // b_tile_size[1]),
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        / 448.0
    )

    triton_result = create_triton_fp8_result(
        input_tensor,
        weight_tensor,
        weight_scales,
        a_tile_size,
        b_tile_size,
    )

    max_result = create_max_fp8_result(
        gpu_session,
        input_tensor,
        weight_tensor,
        weight_scales,
        a_tile_size,
        b_tile_size,
        input_scale_spec=Float8InputScaleSpec(
            granularity=input_scale_granularity,
            origin=Float8ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=a_tile_size,
        ),
        weight_scale_spec=Float8WeightScaleSpec(
            granularity=weight_scale_granularity,
            dtype=DType.float32,
            block_size=b_tile_size,
        ),
    )

    torch.testing.assert_close(
        triton_result.to(max_result.dtype),
        max_result,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
