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
"""Tests Laguna's fused q/k/v attention projection.

The three projections run as one matmul over their row-concatenated weights,
while the checkpoint keeps them as three tensors. These cases pin that each
per-head slice of the fused output equals its own projection, and that the
weights still load under the checkpoint's names.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.laguna_modulev3.layers.attention import (
    LagunaAttention,
)
from max.tensor import Tensor, default_dtype

_HIDDEN = 16
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_ROWS = 3


class _Projection(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, gating_mode: str) -> None:
        super().__init__()
        self.self_attn = LagunaAttention(
            num_attention_heads=_HEADS,
            num_key_value_heads=_KV_HEADS,
            hidden_size=_HIDDEN,
            kv_params=MHAKVCacheParams(
                dtype=DType.float32,
                n_kv_heads=_KV_HEADS,
                head_dim=_HEAD_DIM,
                num_layers=1,
                devices=[DeviceRef.CPU()],
            ),
            is_sliding=False,
            local_window_size=-1,
            gating_mode=gating_mode,
            rotary_dim=_HEAD_DIM // 2,
            scale=1.0,
            dtype=DType.float32,
            norm_dtype=DType.float32,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.self_attn.project_qkv(x)


def _projection(gating_mode: str) -> _Projection:
    with F.lazy(), default_dtype(DType.float32):
        module = _Projection(gating_mode)
        module.to(CPU())
    return module


def _checkpoint(gating_mode: str) -> dict[str, torch.Tensor]:
    """Random weights under the names the checkpoint gives them."""
    q_dim = _HEADS * _HEAD_DIM
    kv_dim = _KV_HEADS * _HEAD_DIM
    gate_dim = _HEADS if gating_mode == "per-head" else q_dim
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    return {
        "self_attn.q_proj.weight": randn(q_dim, _HIDDEN),
        "self_attn.k_proj.weight": randn(kv_dim, _HIDDEN),
        "self_attn.v_proj.weight": randn(kv_dim, _HIDDEN),
        "self_attn.o_proj.weight": randn(_HIDDEN, q_dim),
        "self_attn.g_proj.weight": randn(gate_dim, _HIDDEN),
        "self_attn.q_norm.weight": randn(_HEAD_DIM),
        "self_attn.k_norm.weight": randn(_HEAD_DIM),
    }


def _input_type() -> TensorType:
    return TensorType(DType.float32, (_ROWS, _HIDDEN), device=DeviceRef.CPU())


@pytest.mark.parametrize("gating_mode", ["per-head", "per-element"])
def test_weights_keep_their_checkpoint_names(gating_mode: str) -> None:
    names = {name for name, _ in _projection(gating_mode).parameters}
    assert names == set(_checkpoint(gating_mode))


@pytest.mark.parametrize("gating_mode", ["per-head", "per-element"])
def test_each_slice_equals_its_own_projection(gating_mode: str) -> None:
    weights = _checkpoint(gating_mode)
    compiled = _projection(gating_mode).compile(_input_type(), weights=weights)
    x = torch.randn(
        (_ROWS, _HIDDEN), generator=torch.Generator().manual_seed(1)
    )
    outputs = compiled.execute_raw(Buffer.from_dlpack(x))
    for name, output in zip("qkv", outputs, strict=True):
        expected = x @ weights[f"self_attn.{name}_proj.weight"].T
        np.testing.assert_allclose(
            torch.from_dlpack(output).numpy(),
            expected.reshape((_ROWS, -1, _HEAD_DIM)).numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{name}_proj",
        )


def test_q_k_and_v_run_as_one_matmul() -> None:
    staged, _ = _projection("per-head").stage(
        _input_type(), weights=_checkpoint("per-head")
    )
    matmuls = [line for line in str(staged).splitlines() if "matmul" in line]
    assert len(matmuls) == 1, "\n".join(matmuls)
