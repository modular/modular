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
"""CPU reference tests for the Engram signed-sqrt gate.

The hash+gather op is GPU-only, so these tests pin the gate the hash
feeds: the per-copy key/value split, the Hadamard ``q*k`` weight, the
per-(token, copy) RMS normalization, the ``< 0`` (not ``sign``) branch at
``dot == 0``, the 1e-6 magnitude clamp and the dead-row
zeroing. The oracle is an independent NumPy transcription of the
reference gate math.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.engram import (
    _GATE_CLAMP,
    Engram,
)


def _to_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


_HIDDEN = 128
_HC = 4
_EPS = 1e-20


def _make_engram(q_bf: np.ndarray, k_bf: np.ndarray) -> Engram:
    layout = EngramLayout(
        layer_ids=(1, 5),
        num_embeddings=(0, 0),
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        compressed_vocab_size=100,
        pad_token_id=2,
        vocab_size=20_000,
    )
    engram = Engram(
        layer_ids=(1, 5),
        num_embeddings=tuple(int(p.sum()) for p in layout.primes),
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        compressed_vocab_size=100,
        engram_vocab_size=20_000,
        hidden_size=_HIDDEN,
        hc_mult=_HC,
        pad_token_id=2,
        rms_norm_eps=_EPS,
        device=CPU(),
        token_map=np.arange(50, dtype=np.int32),
    )
    engram.q_weight = F.cast(Tensor.from_dlpack(q_bf), DType.bfloat16)
    engram.k_weight = F.cast(Tensor.from_dlpack(k_bf), DType.bfloat16)
    return engram


def _gate_oracle(
    hidden_bf: np.ndarray,
    kv_bf: np.ndarray,
    q_bf: np.ndarray,
    k_bf: np.ndarray,
    dead: np.ndarray,
) -> np.ndarray:
    total = kv_bf.shape[0]
    key = (
        kv_bf[:, : _HC * _HIDDEN]
        .reshape(total, _HC, _HIDDEN)
        .astype(np.float32)
    )
    value = kv_bf[:, _HC * _HIDDEN :].astype(np.float32)
    hidden = hidden_bf.astype(np.float32)
    weight = q_bf.astype(np.float32) * k_bf.astype(np.float32)

    dot = np.sum(hidden * weight[None] * key, axis=-1)
    hidden_rms = 1.0 / np.sqrt(np.mean(hidden * hidden, axis=-1) + _EPS)
    key_rms = 1.0 / np.sqrt(np.mean(key * key, axis=-1) + _EPS)
    dot = dot * hidden_rms * key_rms * (float(_HIDDEN) ** -0.5)

    mag = np.sqrt(np.maximum(np.abs(dot), _GATE_CLAMP))
    gate = 1.0 / (1.0 + np.exp(-np.where(dot < 0, -mag, mag)))
    alive = (dead == 0).astype(np.float32)[:, None]
    out = hidden + (gate * alive)[:, :, None] * value[:, None, :]
    return _to_bf16(out)


def _run(
    engram: Engram, hidden_bf: np.ndarray, kv_bf: np.ndarray, dead: np.ndarray
) -> np.ndarray:
    hidden_t = F.cast(Tensor.from_dlpack(hidden_bf), DType.bfloat16)
    kv_t = F.cast(Tensor.from_dlpack(kv_bf), DType.bfloat16)
    out = engram._gate(
        hidden_t, kv_t, Tensor.from_dlpack(dead.astype(np.uint8))
    )
    assert out.dtype == DType.bfloat16
    return F.cast(out, DType.float32).to_numpy()


def test_gate_reference_behavior() -> None:
    """Production gate math: keys per copy, shared value, per-copy RMS."""
    rng = np.random.default_rng(18)
    total = 6
    hidden = (rng.standard_normal((total, _HC, _HIDDEN)) * 0.4).astype(
        np.float32
    )
    kv = (rng.standard_normal((total, (_HC + 1) * _HIDDEN)) * 0.4).astype(
        np.float32
    )
    q = (rng.standard_normal((_HC, _HIDDEN)) * 0.3).astype(np.float32)
    k = (rng.standard_normal((_HC, _HIDDEN)) * 0.3).astype(np.float32)
    dead = np.array([0, 0, 0, 0, 0, 0], np.uint8)
    hidden_bf, kv_bf, q_bf, k_bf = map(_to_bf16, (hidden, kv, q, k))

    engram = _make_engram(q_bf, k_bf)
    got = _run(engram, hidden_bf, kv_bf, dead)
    np.testing.assert_array_equal(
        got, _gate_oracle(hidden_bf, kv_bf, q_bf, k_bf, dead)
    )


def test_gate_dead_rows_zeroed() -> None:
    """Dead positions keep the stream exactly (gate forced to 0)."""
    rng = np.random.default_rng(19)
    total = 4
    hidden = (rng.standard_normal((total, _HC, _HIDDEN)) * 0.4).astype(
        np.float32
    )
    kv = (rng.standard_normal((total, (_HC + 1) * _HIDDEN)) * 2.0).astype(
        np.float32
    )
    q = np.ones((_HC, _HIDDEN), np.float32)
    k = np.ones((_HC, _HIDDEN), np.float32)
    dead = np.array([0, 1, 0, 1], np.uint8)
    hidden_bf, kv_bf, q_bf, k_bf = map(_to_bf16, (hidden, kv, q, k))

    engram = _make_engram(q_bf, k_bf)
    got = _run(engram, hidden_bf, kv_bf, dead)
    np.testing.assert_array_equal(
        got, _gate_oracle(hidden_bf, kv_bf, q_bf, k_bf, dead)
    )
    # Dead rows are bit-identical to their input streams.
    np.testing.assert_array_equal(got[1], hidden_bf[1])
    np.testing.assert_array_equal(got[3], hidden_bf[3])


def test_gate_zero_dot_takes_positive_branch() -> None:
    """``dot == 0`` gates at ``+sqrt(1e-6)``, never at sigmoid(0).

    A zero key row gives ``dot == 0`` for every copy; the gate is
    ``sigmoid(+1e-3) > 1/2``, where ``sign``-based gating would give
    exactly 1/2.
    """
    rng = np.random.default_rng(20)
    total = 2
    hidden = (rng.standard_normal((total, _HC, _HIDDEN)) * 0.4).astype(
        np.float32
    )
    kv = np.zeros((total, (_HC + 1) * _HIDDEN), np.float32)
    kv[:, _HC * _HIDDEN :] = 1.0  # the shared value is live
    q = np.ones((_HC, _HIDDEN), np.float32)
    k = np.ones((_HC, _HIDDEN), np.float32)
    dead = np.zeros(total, np.uint8)
    hidden_bf, kv_bf, q_bf, k_bf = map(_to_bf16, (hidden, kv, q, k))

    engram = _make_engram(q_bf, k_bf)
    got = _run(engram, hidden_bf, kv_bf, dead)
    np.testing.assert_array_equal(
        got, _gate_oracle(hidden_bf, kv_bf, q_bf, k_bf, dead)
    )
    # The gate is sigmoid(+sqrt(1e-6)) > 1/2: the positive branch, where a
    # sign-based gate would sit at exactly 1/2.
    expected_gate = 1.0 / (1.0 + np.exp(-np.sqrt(_GATE_CLAMP)))
    assert expected_gate > 0.5
    assert (got > hidden_bf).all()
