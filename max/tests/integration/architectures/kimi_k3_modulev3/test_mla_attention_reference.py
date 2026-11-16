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
"""Checks :class:`KimiK3MLAAttention` against a float64 numpy reference.

Tagged ``manual``: it gates on a numerical tolerance. The padded forward,
run on a q_a/kv_a/gate weight fused by the adapter, must match the
checkpoint's unfused math, which pins the fused input GEMM's split.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    fuse_qkv_a_g_for_tp,
)
from max.tensor import Tensor, default_device

_HIDDEN = 64
_NUM_HEADS = 4
_QK_NOPE = 16
_QK_ROPE = 8
_V_HEAD = 16
_Q_LORA = 32
_KV_LORA = 24


def _make_layer() -> KimiK3MLAAttention:
    return KimiK3MLAAttention(
        hidden_size=_HIDDEN,
        num_heads=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        v_head_dim=_V_HEAD,
        q_lora_rank=_Q_LORA,
        kv_lora_rank=_KV_LORA,
    )


def _rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6) * gamma


@pytest.mark.compile_on_gpu(
    "attention.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_forward_matches_unfused_float64_reference() -> None:
    """The fused q_a/kv_a/gate GEMM reproduces the three separate ones.

    The layer runs in float32 on a weight fused by the adapter's
    :func:`fuse_qkv_a_g_for_tp`. The reference is the checkpoint's unfused
    math in float64: its own q_a, kv_a and gate projections of the layer
    input, the sigmoid gate on the attention output before ``o_proj``. A
    split that handed any block the wrong rows would miss by O(1), far
    above float32 rounding.
    """
    rng = np.random.default_rng(3)
    batch, seq_len = 2, 7
    q_head_dim = _QK_NOPE + _QK_ROPE

    def arr(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(
            np.float32
        )

    w = {
        "q_a_proj": arr(_Q_LORA, _HIDDEN),
        "q_b_proj": arr(_NUM_HEADS * q_head_dim, _Q_LORA),
        "kv_a_proj_with_mqa": arr(_KV_LORA + _QK_ROPE, _HIDDEN),
        "kv_b_proj": arr(_NUM_HEADS * (_QK_NOPE + _V_HEAD), _KV_LORA),
        "g_proj": arr(_NUM_HEADS * _V_HEAD, _HIDDEN),
        "o_proj": arr(_HIDDEN, _NUM_HEADS * _V_HEAD),
    }
    q_gamma = (1 + 0.1 * rng.standard_normal(_Q_LORA)).astype(np.float32)
    kv_gamma = (1 + 0.1 * rng.standard_normal(_KV_LORA)).astype(np.float32)
    x = rng.standard_normal((batch, seq_len, _HIDDEN)).astype(np.float32)

    with default_device(Accelerator()):
        layer = _make_layer()
        layer.qkv_a_g_proj.weight = Tensor(
            fuse_qkv_a_g_for_tp(
                w["q_a_proj"], w["kv_a_proj_with_mqa"], w["g_proj"], 1
            ),
            dtype=DType.float32,
        )
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            getattr(layer, name).weight = Tensor(w[name], dtype=DType.float32)
        layer.q_a_layernorm.weight = Tensor(q_gamma, dtype=DType.float32)
        layer.kv_a_layernorm.weight = Tensor(kv_gamma, dtype=DType.float32)
        got = layer(Tensor(x, dtype=DType.float32)).to_numpy()

    w64 = {name: weight.astype(np.float64) for name, weight in w.items()}
    x64 = x.astype(np.float64)
    q = _rms_norm(x64 @ w64["q_a_proj"].T, q_gamma) @ w64["q_b_proj"].T
    q = q.reshape(batch, seq_len, _NUM_HEADS, q_head_dim)
    kv_a = x64 @ w64["kv_a_proj_with_mqa"].T
    kv = _rms_norm(kv_a[..., :_KV_LORA], kv_gamma) @ w64["kv_b_proj"].T
    kv = kv.reshape(batch, seq_len, _NUM_HEADS, _QK_NOPE + _V_HEAD)
    k_rot = np.broadcast_to(
        kv_a[..., None, _KV_LORA:], (batch, seq_len, _NUM_HEADS, _QK_ROPE)
    )
    k = np.concatenate([kv[..., :_QK_NOPE], k_rot], axis=-1)
    scores = np.einsum("bshd,bthd->bhst", q, k) * q_head_dim**-0.5
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    scores = np.where(causal, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    attn = np.einsum("bhst,bthd->bshd", probs, kv[..., _QK_NOPE:])
    gate = 1.0 / (1.0 + np.exp(-(x64 @ w64["g_proj"].T)))
    want = (attn.reshape(batch, seq_len, -1) * gate) @ w64["o_proj"].T

    err = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
    print(f"fused MLA vs unfused float64 reference: rel_error={err}")
    # Leaves room for a TF32 float32 matmul; swapping the q_a and kv_a
    # blocks measured 1.24 here.
    assert err < 1e-2, f"fused MLA diverges from the unfused reference: {err}"
