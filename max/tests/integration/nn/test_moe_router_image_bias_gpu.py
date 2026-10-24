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
"""Tests the router GEMM's float32 result and the single-group router that
picks each token's selection bias from its id."""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kernels import moe_router_image_bias
from torch.utils.dlpack import from_dlpack

N_EXPERTS = 384
TOPK = 6
SCALE = 1.5
HIDDEN = 1024
SENTINEL = 7


def _gpu(dtype: DType, *shape: int | str) -> TensorType:
    return TensorType(dtype, list(shape), device=DeviceRef.GPU())


def _sqrt_softplus(x: TensorValue) -> TensorValue:
    zero = ops.constant(0.0, DType.float32, x.device)
    return ops.sqrt(ops.max(x, zero) + ops.log1p(ops.exp(-ops.abs(x))))


def _logits(x: TensorValue, w: TensorValue) -> TensorValue:
    return ops.matmul(x, ops.transpose(w, 0, 1), out_dtype=DType.float32)


@pytest.mark.parametrize("hidden", [HIDDEN, 5120])
@pytest.mark.parametrize("m", [1, 7, 144, 577])
def test_router_logits_float32(
    gpu_session: InferenceSession, m: int, hidden: int
) -> None:
    with Graph(
        "router_logits_float32",
        input_types=(
            _gpu(DType.bfloat16, "m", hidden),
            _gpu(DType.bfloat16, N_EXPERTS, hidden),
        ),
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(_logits(a, b))
    compiled = gpu_session.load(graph)

    torch.manual_seed(m)
    a = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(N_EXPERTS, hidden, device="cuda") * hidden**-0.5).to(
        torch.bfloat16
    )
    (got,) = (from_dlpack(t) for t in compiled.execute(a, b))
    assert got.dtype == torch.float32

    want = a.double() @ b.double().T
    err = (got.double() - want).abs().max().item()
    # One bf16 rounding of the output alone would be ~4e-3 relative; the
    # float32 accumulator stays orders of magnitude below that.
    bf16_err = (want.to(torch.bfloat16).double() - want).abs().max().item()
    assert err < 1e-5 * want.abs().max().item(), err
    assert err < bf16_err / 100, (err, bf16_err)


def _gate_graph() -> Graph:
    """The DeepSeek-V4.1 gate: float32 logits, sqrt-softplus scores, and
    `bias_vl` instead of `bias` on image-sentinel tokens."""
    with Graph(
        "image_bias_gate",
        input_types=(
            _gpu(DType.bfloat16, "m", HIDDEN),
            _gpu(DType.int64, "m"),
            _gpu(DType.bfloat16, N_EXPERTS, HIDDEN),
            _gpu(DType.float32, N_EXPERTS),
            _gpu(DType.float32, N_EXPERTS),
        ),
    ) as graph:
        x, tid, w, bias, bias_vl = (v.tensor for v in graph.inputs)
        indices, weights = moe_router_image_bias(
            _sqrt_softplus(_logits(x, w)),
            bias,
            bias_vl,
            tid,
            image_token_id=SENTINEL,
            n_routed_experts=N_EXPERTS,
            n_experts_per_tok=TOPK,
            routed_scaling_factor=SCALE,
        )
        graph.output(indices, weights)
    return graph


def _check_routing(
    indices: torch.Tensor,
    weights: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    """`indices` is a top-k of `scores + bias` per row (a near-tie may pick
    either side), and `weights` are its unbiased scores, normalized with a
    floored sum and scaled."""
    indices = indices.long()
    assert bool((indices.sort(-1).values.diff(dim=-1) > 0).all())
    biased = scores + bias
    chosen = torch.zeros_like(biased, dtype=torch.bool)
    chosen.scatter_(-1, indices, True)
    worst_in = biased.gather(-1, indices).min(-1).values
    best_out = biased.masked_fill(chosen, float("-inf")).max(-1).values
    assert bool((worst_in >= best_out - 1e-5).all())
    picked = scores.gather(-1, indices)
    total = picked.sum(-1, keepdim=True)
    want = SCALE * picked / torch.where(total > 0, total, 1.0)
    torch.testing.assert_close(weights.double(), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("m", [1, 9, 144, 577, 4099])
def test_router_image_bias(gpu_session: InferenceSession, m: int) -> None:
    compiled = gpu_session.load(_gate_graph())

    torch.manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(N_EXPERTS, HIDDEN, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    # Each bias lifts its own 12 experts, so a token's scores pick 6 of
    # them and a token that took the wrong bias lands in the wrong 12.
    lifted = 2 * TOPK
    bias = torch.zeros(N_EXPERTS, device="cuda")
    bias[:lifted] = 3.0
    bias_vl = torch.zeros(N_EXPERTS, device="cuda")
    bias_vl[-lifted:] = 3.0
    tid = torch.randint(0, 100, (m,), device="cuda", dtype=torch.int64)
    tid[::2] = SENTINEL

    indices, weights = (
        from_dlpack(t) for t in compiled.execute(x, tid, w, bias, bias_vl)
    )
    image = tid == SENTINEL
    scores = torch.nn.functional.softplus(x.double() @ w.double().T).sqrt()
    row_bias = torch.where(image[:, None], bias_vl, bias).double()
    _check_routing(indices, weights, scores, row_bias)
    assert bool((indices[image] >= N_EXPERTS - lifted).all())
    assert bool((indices[~image] < lifted).all())


def test_router_all_zero_selection(gpu_session: InferenceSession) -> None:
    """A token whose selected scores are all zero keeps zero weights."""
    with Graph(
        "zero_selection",
        input_types=(
            _gpu(DType.float32, "m", N_EXPERTS),
            _gpu(DType.float32, N_EXPERTS),
            _gpu(DType.float32, N_EXPERTS),
            _gpu(DType.int64, "m"),
        ),
    ) as graph:
        scores, bias, bias_vl, tid = (v.tensor for v in graph.inputs)
        graph.output(
            *moe_router_image_bias(
                scores,
                bias,
                bias_vl,
                tid,
                image_token_id=SENTINEL,
                n_routed_experts=N_EXPERTS,
                n_experts_per_tok=TOPK,
                routed_scaling_factor=SCALE,
            )
        )
    compiled = gpu_session.load(graph)

    scores = torch.rand(4, N_EXPERTS, device="cuda")
    scores[:2] = 0.0
    bias = torch.zeros(N_EXPERTS, device="cuda")
    bias[:TOPK] = 1.0
    bias_vl = torch.zeros(N_EXPERTS, device="cuda")
    bias_vl[-TOPK:] = 1.0
    tid = torch.tensor([0, SENTINEL, 0, SENTINEL], device="cuda")
    indices, weights = (
        from_dlpack(t) for t in compiled.execute(scores, bias, bias_vl, tid)
    )
    assert bool(torch.isfinite(weights).all())
    row_bias = torch.where((tid == SENTINEL)[:, None], bias_vl, bias)
    _check_routing(indices, weights, scores.double(), row_bias.double())
    torch.testing.assert_close(weights[:2], torch.zeros(2, TOPK, device="cuda"))
