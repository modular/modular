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

"""Tests the grouped matmul with an elementwise epilogue fused into it."""

import re

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from torch.utils.dlpack import from_dlpack

NUM_EXPERTS = 8
K = 256
N = 192


def _build(epilogue: str, escaping: bool) -> Graph:
    """Builds the routed up projection followed by an elementwise epilogue.

    With ``escaping`` the matmul output is also a graph output, so the fused
    epilogue stores it beside the epilogue's result.
    """
    with Graph(
        f"grouped_matmul_{epilogue}",
        input_types=(
            TensorType(DType.int32, ["tokens"], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, ["tokens", K], device=DeviceRef.GPU()),
            TensorType(
                DType.bfloat16, [NUM_EXPERTS, N, K], device=DeviceRef.GPU()
            ),
            TensorType(DType.bfloat16, [N], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, ["tokens", 1], device=DeviceRef.GPU()),
        ),
    ) as g:
        topk_ids, x, weight, bias, row_scale = (v.tensor for v in g.inputs)
        (
            token_expert_order,
            expert_start_indices,
            _,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(topk_ids, NUM_EXPERTS)
        permuted = ops.gather(x, token_expert_order, axis=0)
        up = grouped_matmul_ragged(
            permuted,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        if epilogue == "relu2":
            relu = ops.relu(up)
            act = relu * relu
        else:
            # Reads other tensors at the output coordinates, so a wrong row
            # or column in the fused epilogue changes the result.
            act = ops.relu(up + bias) * ops.gather(
                row_scale, token_expert_order, axis=0
            )
        outputs = [act, token_expert_order]
        if escaping:
            outputs.append(up)
        g.output(*outputs)
    return g


# The epilogue's elementwise ops, in the order a kernel summary lists them.
_EPILOGUE_OPS = {
    "relu2": ["mo.relu", "mo.mul"],
    "bias_row_scale": ["mo.add", "mo.relu", "mo.mul"],
}


def _fuses_whole_epilogue(model: Model, epilogue: str) -> bool:
    """Whether one kernel runs the grouped matmul and every epilogue op, and
    no other kernel runs any of them."""
    ops = _EPILOGUE_OPS[epilogue]
    chain = re.compile(
        r"mo\.grouped\.matmul\.ragged.*" + ".*".join(map(re.escape, ops))
    )
    summaries = model.kernel_summaries
    others = [s for s in summaries if "mo.grouped.matmul.ragged" not in s]
    return any(chain.search(s) for s in summaries) and not any(
        op in s for s in others for op in ops
    )


@pytest.mark.parametrize("epilogue", ["relu2", "bias_row_scale"])
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 300])
def test_grouped_matmul_fused_epilogue(epilogue: str, num_tokens: int) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    fused = session.load(_build(epilogue, escaping=False))
    escaping = session.load(_build(epilogue, escaping=True))
    # The comparisons below pass even if fusion stops happening, so check the
    # compiled kernels too.
    assert _fuses_whole_epilogue(fused, epilogue), fused.kernel_summaries
    assert _fuses_whole_epilogue(escaping, epilogue), escaping.kernel_summaries

    # Unseeded data occasionally put an output on the relu boundary, where
    # bf16 rounding flips it past the tolerance.
    torch.manual_seed(num_tokens)
    rng = np.random.default_rng(num_tokens)
    topk_ids = rng.integers(0, NUM_EXPERTS, size=num_tokens, dtype=np.int32)
    x = torch.randn(num_tokens, K, dtype=torch.bfloat16)
    weight = torch.randn(NUM_EXPERTS, N, K, dtype=torch.bfloat16) / K**0.5
    bias = torch.randn(N, dtype=torch.bfloat16)
    row_scale = torch.randn(num_tokens, 1, dtype=torch.bfloat16)
    inputs = [
        Buffer.from_numpy(topk_ids).to(device),
        Buffer.from_dlpack(x).to(device),
        Buffer.from_dlpack(weight).to(device),
        Buffer.from_dlpack(bias).to(device),
        Buffer.from_dlpack(row_scale).to(device),
    ]

    out, order = fused.execute(*inputs)[:2]
    out_t = from_dlpack(out).cpu()
    order_np = from_dlpack(order).cpu().numpy().astype(np.int64)

    # Storing the matmul output too must not change a single bit of the
    # result. The two runs may order rows within an expert differently, so
    # compare the rows by token.
    escaping_out, escaping_order, escaping_up = escaping.execute(*inputs)
    escaping_order_np = (
        from_dlpack(escaping_order).cpu().numpy().astype(np.int64)
    )
    by_token = torch.empty_like(out_t)
    by_token[torch.from_numpy(order_np)] = out_t
    escaping_by_token = torch.empty_like(out_t)
    escaping_by_token[torch.from_numpy(escaping_order_np)] = from_dlpack(
        escaping_out
    ).cpu()
    torch.testing.assert_close(by_token, escaping_by_token, rtol=0, atol=0)

    # moe_create_indices does not fix the row order within an expert, so the
    # reference is built from the order the graph returns. Row r of the
    # output is token order[r] through its expert's weights.
    rows = x.float()[order_np]
    experts = weight.float()[topk_ids[order_np]]
    up = torch.einsum("rk,rnk->rn", rows, experts)
    if epilogue == "relu2":
        ref = torch.relu(up) ** 2
    else:
        ref = torch.relu(up + bias.float()) * row_scale.float()[order_np]
    torch.testing.assert_close(out_t.float(), ref, rtol=2e-2, atol=2e-2)

    # The matmul output the epilogue stores, in the escaping run's row order.
    escaping_rows = x.float()[escaping_order_np]
    escaping_experts = weight.float()[topk_ids[escaping_order_np]]
    escaping_ref = torch.einsum("rk,rnk->rn", escaping_rows, escaping_experts)
    torch.testing.assert_close(
        from_dlpack(escaping_up).cpu().float(),
        escaping_ref,
        rtol=2e-2,
        atol=2e-2,
    )
