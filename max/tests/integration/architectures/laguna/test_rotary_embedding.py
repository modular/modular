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
"""Laguna's partial rotary is split across two pieces that only agree together.

``LagunaYarnPartialRotaryEmbedding`` widens the YaRN table to ``head_dim`` with
identity pairs, and ``LagunaAttention`` permutes q and k lanes so the fused
store's ``(i, i + head_dim // 2)`` pairing lands on the reference's
``(j, j + rotary_dim // 2)``. Either one alone is wrong, and both produce
plausible-looking finite logits, so the composition is checked end to end
against the reference rather than either half being checked on its own.

The reference is the pinned transformers' own ``laguna`` implementation, driven
from the committed ``testdata/s21_config.json`` -- not a hand port, which would
only prove the port and the architecture agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max.driver import CPU
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.rotary_embedding import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.laguna.layers.rotary_embedding import (
    LagunaRotaryEmbedding,
    LagunaYarnPartialRotaryEmbedding,
)
from transformers.models.laguna.configuration_laguna import (
    LagunaConfig as HFLagunaConfig,
)
from transformers.models.laguna.modeling_laguna import (
    LagunaRotaryEmbedding as HFLagunaRotaryEmbedding,
)
from transformers.models.laguna.modeling_laguna import apply_rotary_pos_emb

TESTDATA = Path(__file__).parent / "testdata"

HIDDEN_SIZE = 3072
HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
SEQ_LEN = 16
MAX_SEQ_LEN = 64


def _hf_config() -> HFLagunaConfig:
    with (TESTDATA / "s21_config.json").open() as f:
        return HFLagunaConfig(**json.load(f))


def _rope_params(layer_type: str) -> dict[str, Any]:
    return dict(_hf_config().rope_parameters[layer_type])


def _evaluate_freqs_cis(rope: RotaryEmbedding) -> np.ndarray:
    """Materialises a RoPE table through a CPU graph.

    Args:
        rope: The rotary embedding whose table to evaluate.

    Returns:
        The ``[positions, head_dim]`` table the fused store would read.
    """
    session = InferenceSession(devices=[CPU()])
    with Graph("laguna_freqs_cis", input_types=[]) as graph:
        graph.output(rope.freqs_cis)
    compiled = session.load(graph, weights_registry={})
    (buf,) = compiled.execute()
    return torch.from_dlpack(buf).float().numpy()


def _lane_permutation(rotary_dim: int, head_dim: int) -> np.ndarray:
    """Returns the lane order ``LagunaAttention._permute_rotary_lanes`` produces."""
    half_rotated = rotary_dim // 2
    half_passed = (head_dim - rotary_dim) // 2
    return np.concatenate(
        [
            np.arange(0, half_rotated),
            np.arange(rotary_dim, rotary_dim + half_passed),
            np.arange(half_rotated, rotary_dim),
            np.arange(rotary_dim + half_passed, head_dim),
        ]
    )


def _kernel_rope(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Applies the fused store's non-interleaved rotation in numpy.

    Mirrors ``rope_value`` + ``get_safetensors_idx`` in
    ``max/kernels/src/nn/rope_split_store.mojo``: pair index ``j`` of the flat
    ``[positions, head_dim]`` table holds ``(cos, sin)`` at ``(2j, 2j + 1)``
    and rotates lanes ``(j, j + head_dim // 2)``.

    Args:
        x: ``[seq, heads, head_dim]`` activations in kernel lane order.
        table: The ``[positions, head_dim]`` table, sliced to ``seq`` rows.

    Returns:
        The rotated activations, same shape as ``x``.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    cos = table[:, 0::2][:, None, :]
    sin = table[:, 1::2][:, None, :]
    re, im = x[..., :half], x[..., half:]
    return np.concatenate([re * cos - im * sin, re * sin + im * cos], axis=-1)


def _max_rotation(
    table: np.ndarray, x: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """Runs MAX's full partial-rotary path: permute, rotate, un-permute."""
    perm = _lane_permutation(rotary_dim, x.shape[-1])
    rotated = _kernel_rope(x[..., perm], table[: x.shape[0]])
    out = np.empty_like(rotated)
    out[..., perm] = rotated
    return out


def _hf_rotation(
    layer_type: str, q: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotates q and k with the native transformers Laguna implementation."""
    config = _hf_config()
    rotary = HFLagunaRotaryEmbedding(config)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long)[None, :]
    # [seq, heads, head_dim] -> the [batch, heads, seq, head_dim] HF wants.
    q_t = torch.from_numpy(q).permute(1, 0, 2)[None]
    k_t = torch.from_numpy(k).permute(1, 0, 2)[None]
    cos, sin = rotary(q_t, position_ids, layer_type=layer_type)
    q_out, k_out = apply_rotary_pos_emb(q_t, k_t, cos, sin)
    return (
        q_out[0].permute(1, 0, 2).numpy(),
        k_out[0].permute(1, 0, 2).numpy(),
    )


def _random_qk() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((SEQ_LEN, NUM_Q_HEADS, HEAD_DIM), dtype=np.float32)
    k = rng.standard_normal((SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32)
    return q, k


def _full_attention_rope() -> LagunaYarnPartialRotaryEmbedding:
    params = _rope_params("full_attention")
    return LagunaYarnPartialRotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=NUM_Q_HEADS,
        theta=float(params["rope_theta"]),
        max_seq_len=MAX_SEQ_LEN,
        head_dim=HEAD_DIM,
        interleaved=False,
        scaling_params=YarnScalingParams(
            factor=float(params["factor"]),
            beta_fast=float(params["beta_fast"]),
            beta_slow=float(params["beta_slow"]),
            original_max_position_embeddings=int(
                params["original_max_position_embeddings"]
            ),
            truncate=True,
        ),
        partial_rotary_factor=float(params["partial_rotary_factor"]),
    )


def _sliding_attention_rope() -> LagunaRotaryEmbedding:
    params = _rope_params("sliding_attention")
    return LagunaRotaryEmbedding(
        dim=HIDDEN_SIZE,
        n_heads=NUM_Q_HEADS,
        theta=float(params["rope_theta"]),
        max_seq_len=MAX_SEQ_LEN,
        head_dim=HEAD_DIM,
        interleaved=False,
        scaling_params=ProportionalScalingParams(
            partial_rotary_factor=float(params["partial_rotary_factor"])
        ),
    )


def test_partial_yarn_rotation_matches_transformers() -> None:
    params = _rope_params("full_attention")
    rotary_dim = int(HEAD_DIM * params["partial_rotary_factor"])
    assert rotary_dim < HEAD_DIM

    q, k = _random_qk()
    q_hf, k_hf = _hf_rotation("full_attention", q, k)
    table = _evaluate_freqs_cis(_full_attention_rope())

    np.testing.assert_allclose(
        _max_rotation(table, q, rotary_dim), q_hf, atol=1e-4
    )
    np.testing.assert_allclose(
        _max_rotation(table, k, rotary_dim), k_hf, atol=1e-4
    )


def test_pass_through_lanes_are_left_alone() -> None:
    """The lanes past ``rotary_dim`` must survive the kernel bit-for-bit."""
    params = _rope_params("full_attention")
    rotary_dim = int(HEAD_DIM * params["partial_rotary_factor"])
    q, _ = _random_qk()

    table = _evaluate_freqs_cis(_full_attention_rope())
    out = _max_rotation(table, q, rotary_dim)
    np.testing.assert_array_equal(out[..., rotary_dim:], q[..., rotary_dim:])


def test_the_lane_permutation_preserves_attention_scores() -> None:
    """``q . k`` per head is what attention sees, and a shared permutation
    cannot change it -- which is why ``o_proj`` needs no un-permute."""
    q, k = _random_qk()
    perm = _lane_permutation(64, HEAD_DIM)
    # One KV head per group of query heads; compare like for like.
    k_grouped = np.repeat(k, NUM_Q_HEADS // NUM_KV_HEADS, axis=1).astype(
        np.float64
    )
    q = q.astype(np.float64)

    before = np.einsum("thd,thd->th", q, k_grouped)
    after = np.einsum("thd,thd->th", q[..., perm], k_grouped[..., perm])
    # Only the float64 summation order differs, not the value.
    np.testing.assert_allclose(before, after, rtol=1e-12)


def test_full_rotary_yarn_table_matches_the_unmodified_base_class() -> None:
    """``partial_rotary_factor == 1.0`` must not perturb the YaRN table."""
    scaling = YarnScalingParams(
        factor=64.0,
        beta_fast=64.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
        truncate=True,
    )
    kwargs: dict[str, Any] = dict(
        dim=HIDDEN_SIZE,
        n_heads=NUM_Q_HEADS,
        theta=500000.0,
        max_seq_len=MAX_SEQ_LEN,
        head_dim=HEAD_DIM,
        interleaved=False,
        scaling_params=scaling,
    )
    partial = LagunaYarnPartialRotaryEmbedding(
        **kwargs, partial_rotary_factor=1.0
    )
    np.testing.assert_allclose(
        _evaluate_freqs_cis(partial),
        _evaluate_freqs_cis(YarnRotaryEmbedding(**kwargs)),
        atol=1e-6,
    )


def test_sliding_table_is_plain_rotate_half() -> None:
    params = _rope_params("sliding_attention")
    theta = float(params["rope_theta"])
    table = _evaluate_freqs_cis(_sliding_attention_rope())[:SEQ_LEN]

    inv_freq = 1.0 / (theta ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    angles = np.outer(np.arange(SEQ_LEN), inv_freq)
    np.testing.assert_allclose(table[:, 0::2], np.cos(angles), atol=1e-5)
    np.testing.assert_allclose(table[:, 1::2], np.sin(angles), atol=1e-5)


def test_sliding_rotation_matches_transformers() -> None:
    q, k = _random_qk()
    q_hf, k_hf = _hf_rotation("sliding_attention", q, k)
    table = _evaluate_freqs_cis(_sliding_attention_rope())

    np.testing.assert_allclose(
        _max_rotation(table, q, HEAD_DIM), q_hf, atol=1e-4
    )
    np.testing.assert_allclose(
        _max_rotation(table, k, HEAD_DIM), k_hf, atol=1e-4
    )
