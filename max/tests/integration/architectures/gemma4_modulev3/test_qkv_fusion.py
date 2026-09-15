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
"""Tests for the Gemma4 ModuleV3 graph-time QKV weight fusion.

The graph arch pre-fuses Q/K/V in the checkpoint adapter
(``fuse_gemma4_projection_weights``); the ModuleV3 arch keeps the native
per-projection weights and fuses at graph-build time via the
``Gemma4Attention.wqkv`` property. These tests pin the same contract:

- sliding layers fuse ``[q_proj | k_proj | v_proj]`` by row-concat;
- global k==v layers register no ``v_proj`` and fuse ``[q_proj | k_proj]``;
- the forward split feeds V from the raw (pre-norm) K on k==v layers.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor, default_device, default_dtype
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4_modulev3.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.architectures.gemma4_modulev3.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)

# Tiny structural dims (the contract is layout, not scale).
_HIDDEN = 32
_N_HEADS = 4
_SLIDING_HEAD_DIM = 8
_SLIDING_NUM_KV = 2
_GLOBAL_HEAD_DIM = 16
_GLOBAL_NUM_KV = 2
_MAX_SEQ_LEN = 16


def _make_attention(*, is_sliding: bool) -> Gemma4Attention:
    head_dim = _SLIDING_HEAD_DIM if is_sliding else _GLOBAL_HEAD_DIM
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        devices=[DeviceRef.CPU()],
        n_kv_heads=_SLIDING_NUM_KV if is_sliding else _GLOBAL_NUM_KV,
        head_dim=head_dim,
        num_layers=1,
        page_size=128,
    )
    rope_local = RotaryEmbedding(
        dim=_HIDDEN,
        n_heads=_N_HEADS,
        theta=10000.0,
        max_seq_len=_MAX_SEQ_LEN,
        device=CPU(),
        head_dim=_SLIDING_HEAD_DIM,
        interleaved=False,
    )
    rope_global = ProportionalRotaryEmbedding(
        dim=_HIDDEN,
        n_heads=_N_HEADS,
        theta=1000000.0,
        max_seq_len=_MAX_SEQ_LEN,
        device=CPU(),
        head_dim=_GLOBAL_HEAD_DIM,
        interleaved=False,
        scaling_params=ProportionalScalingParams(partial_rotary_factor=0.25),
    )
    return Gemma4Attention(
        rope_global=rope_global,
        rope_local=rope_local,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_SLIDING_NUM_KV,
        num_global_key_value_heads=_GLOBAL_NUM_KV,
        attention_k_eq_v=True,
        hidden_size=_HIDDEN,
        kv_params=kv_params,
        layer_idx_in_cache=0,
        is_sliding=is_sliding,
    )


def _rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _set_weights(
    attn: Gemma4Attention, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Overwrites projection weights with known values; returns them."""
    q_dim, kv_dim = attn.q_weight_dim, attn.kv_weight_dim
    weights = {
        "q_proj": _rand(rng, q_dim, _HIDDEN),
        "k_proj": _rand(rng, kv_dim, _HIDDEN),
    }
    attn.q_proj.weight = Tensor.from_dlpack(weights["q_proj"])
    attn.k_proj.weight = Tensor.from_dlpack(weights["k_proj"])
    if attn._has_v_proj:
        weights["v_proj"] = _rand(rng, kv_dim, _HIDDEN)
        attn.v_proj.weight = Tensor.from_dlpack(weights["v_proj"])
    return weights


# --------------------------------------------------------------------------- #
# Parameter registration
# --------------------------------------------------------------------------- #


def test_sliding_layer_registers_v_proj() -> None:
    """Sliding layers keep the full q/k/v projection set."""
    with F.lazy(), default_device(CPU()):
        attn = _make_attention(is_sliding=True)
    names = {name for name, _ in attn.parameters}
    assert {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    } == names


def test_global_k_eq_v_layer_registers_no_v_proj() -> None:
    """Global k==v layers have no v_proj parameter (nothing to load)."""
    with F.lazy(), default_device(CPU()):
        attn = _make_attention(is_sliding=False)
    names = {name for name, _ in attn.parameters}
    assert "v_proj.weight" not in names
    assert {
        "q_proj.weight",
        "k_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    } == names


# --------------------------------------------------------------------------- #
# Fused weight layout
# --------------------------------------------------------------------------- #


def test_sliding_wqkv_is_row_concat_of_q_k_v() -> None:
    """On a sliding layer wqkv == row-concat [q_proj | k_proj | v_proj]."""
    rng = np.random.default_rng(42)
    with default_device(CPU()), default_dtype(DType.float32):
        attn = _make_attention(is_sliding=True)
        weights = _set_weights(attn, rng)
        wqkv = attn.wqkv.to_numpy()

    expected = np.concatenate(
        [weights["q_proj"], weights["k_proj"], weights["v_proj"]], axis=0
    )
    q_dim, kv_dim = attn.q_weight_dim, attn.kv_weight_dim
    assert wqkv.shape == (q_dim + 2 * kv_dim, _HIDDEN)
    np.testing.assert_array_equal(wqkv, expected)


def test_global_k_eq_v_wqkv_is_row_concat_of_q_k() -> None:
    """On a global k==v layer wqkv == row-concat [q_proj | k_proj]."""
    rng = np.random.default_rng(42)
    with default_device(CPU()), default_dtype(DType.float32):
        attn = _make_attention(is_sliding=False)
        weights = _set_weights(attn, rng)
        wqkv = attn.wqkv.to_numpy()

    expected = np.concatenate([weights["q_proj"], weights["k_proj"]], axis=0)
    q_dim, kv_dim = attn.q_weight_dim, attn.kv_weight_dim
    assert wqkv.shape == (q_dim + kv_dim, _HIDDEN)
    np.testing.assert_array_equal(wqkv, expected)


# --------------------------------------------------------------------------- #
# Fused forward split
# --------------------------------------------------------------------------- #


def test_k_eq_v_split_assigns_v_from_raw_k() -> None:
    """The k==v fused split's last segment is the raw (pre-norm) K.

    Mirrors the forward's split of ``x @ wqkv.T`` (``parts[-1]`` doubles as
    V when there is no v_proj) and checks each segment against the manual
    per-projection matmul. The end-to-end numeric check of the full forward
    (including the v_norm applied to this raw K) lives in test_attention.py.
    """
    rng = np.random.default_rng(7)
    with default_device(CPU()), default_dtype(DType.float32):
        attn = _make_attention(is_sliding=False)
        weights = _set_weights(attn, rng)
        assert not attn._has_v_proj

        x_np = _rand(rng, 3, _HIDDEN)
        x = Tensor.from_dlpack(x_np)
        fused = x @ attn.wqkv.T
        # Same splits the forward uses on k==v layers: [q_dim, kv_dim].
        parts = fused.split([attn.q_weight_dim, attn.kv_weight_dim], axis=-1)
        x_q, x_k, x_v = parts[0], parts[1], parts[-1]

        np.testing.assert_allclose(
            x_q.to_numpy(), x_np @ weights["q_proj"].T, rtol=1e-5, atol=1e-5
        )
        raw_k = x_np @ weights["k_proj"].T
        np.testing.assert_allclose(x_k.to_numpy(), raw_k, rtol=1e-5, atol=1e-5)
        # V is fed from the raw pre-norm K segment.
        np.testing.assert_array_equal(x_v.to_numpy(), x_k.to_numpy())


def test_sliding_split_assigns_v_from_v_proj() -> None:
    """On sliding layers the fused split's V segment comes from v_proj."""
    rng = np.random.default_rng(7)
    with default_device(CPU()), default_dtype(DType.float32):
        attn = _make_attention(is_sliding=True)
        weights = _set_weights(attn, rng)
        assert attn._has_v_proj

        x_np = _rand(rng, 3, _HIDDEN)
        x = Tensor.from_dlpack(x_np)
        fused = x @ attn.wqkv.T
        q_dim, kv_dim = attn.q_weight_dim, attn.kv_weight_dim
        parts = fused.split([q_dim, kv_dim, kv_dim], axis=-1)

        np.testing.assert_allclose(
            parts[-1].to_numpy(),
            x_np @ weights["v_proj"].T,
            rtol=1e-5,
            atol=1e-5,
        )
