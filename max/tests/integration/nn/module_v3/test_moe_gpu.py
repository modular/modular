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
"""Validates the ModuleV3 MoE top-k combine against PyTorch. GPU-only."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.experimental.nn.common_layers.functional_kernels import moe_finalize
from max.experimental.nn.common_layers.moe import MoE
from max.experimental.tensor import (
    Tensor,
    TensorType,
    default_device,
    default_dtype,
)

_HIDDEN = 256
_EXPERTS = 8
_TOP_K = 2
_MOE_DIM = 128


@pytest.mark.parametrize("tokens", [1, 37])
@pytest.mark.parametrize("out_type", [DType.bfloat16, DType.float32])
def test_moe_finalize(tokens: int, out_type: DType) -> None:
    rows = tokens * _TOP_K
    gen = torch.Generator().manual_seed(tokens)
    down = torch.randn(rows, _HIDDEN, generator=gen).to(torch.bfloat16)
    weights = torch.rand(tokens, _TOP_K, generator=gen)
    restore = np.random.default_rng(tokens).permutation(rows).astype(np.uint32)

    device = Accelerator()
    got = moe_finalize(
        Tensor(down.cuda(), device=device),
        Tensor.from_dlpack(restore).to(device),
        Tensor(weights.cuda(), device=device),
        out_type,
    )
    assert got.dtype == out_type
    got_t = torch.from_dlpack(got.to(CPU())).double()

    picked = down.double()[torch.from_numpy(restore.astype(np.int64))]
    want = torch.einsum(
        "tk,tkh->th",
        weights.double(),
        picked.reshape(tokens, _TOP_K, _HIDDEN),
    )
    tol = 1e-2 if out_type == DType.bfloat16 else 1e-5
    torch.testing.assert_close(got_t, want, rtol=tol, atol=tol)


def test_moe_layer() -> None:
    """Runs the whole layer with routing pinned to exact, tie-free scores."""
    tokens = 37
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(tokens, _HIDDEN, generator=gen)
    # The gate reads only the first `_EXPERTS` features, which hold a
    # per-token permutation of distinct bf16-exact values, so MAX and the
    # reference pick the same experts with the same weights.
    for t in range(tokens):
        x[t, :_EXPERTS] = (
            torch.randperm(_EXPERTS, generator=gen).float() + 1
        ) / 8
    x = x.to(torch.bfloat16)
    gate = torch.zeros(_EXPERTS, _HIDDEN)
    gate[:, :_EXPERTS] = torch.eye(_EXPERTS)

    with default_device(CPU()), default_dtype(DType.bfloat16):
        layer = MoE(
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            moe_dim=_MOE_DIM,
        )
    weights = {
        name: (torch.randn([int(d) for d in p.shape], generator=gen) * 0.05).to(
            torch.bfloat16
        )
        for name, p in layer.parameters
    }
    weights["gate.gate_score.weight"] = gate.to(torch.bfloat16)

    device = Accelerator()
    compiled = layer.to(device).compile(
        TensorType(DType.bfloat16, ["tokens", _HIDDEN], device=device),
        weights=weights,
    )
    got = torch.from_dlpack(
        compiled(Tensor(x.cuda(), device=device)).to(CPU())
    ).float()

    w = {name: t.float() for name, t in weights.items()}
    scores = x.float() @ w["gate.gate_score.weight"].T
    top_w, top_e = torch.topk(scores, _TOP_K, dim=-1)
    want = torch.zeros(tokens, _HIDDEN)
    for t in range(tokens):
        for k in range(_TOP_K):
            e = int(top_e[t, k])
            h = torch.nn.functional.silu(
                x[t].float() @ w[f"experts.{e}.gate_proj.weight"].T
            ) * (x[t].float() @ w[f"experts.{e}.up_proj.weight"].T)
            want[t] += top_w[t, k] * (h @ w[f"experts.{e}.down_proj.weight"].T)

    torch.testing.assert_close(
        got, want, rtol=2e-2, atol=2e-2 * float(want.abs().max())
    )
