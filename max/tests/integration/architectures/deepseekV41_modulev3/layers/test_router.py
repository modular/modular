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
"""CPU reference tests for the V4.1 ``sqrtsoftplus`` ``noaux_tc`` router.

Production shape: 5120-wide inputs against the 384-expert bf16 gate, top-6,
scaling 1.5. The oracle is an independent NumPy transcription of the HF
``Gate`` and the vLLM ``dsv4_topk`` selection loop.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.router import (
    Router,
    _sqrt_softplus,
    _stable_top_k,
)
from max.tensor import Tensor

_HIDDEN = 5120
_EXPERTS = 384
_TOPK = 6
_SCALE = 1.5


def _to_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


def _sqrt_softplus_oracle(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    softplus = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    return np.sqrt(softplus).astype(np.float32)


def _select_oracle(
    scores: np.ndarray, bias: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The oracle's loop: max per slot, lowest id among ties, -inf masking.

    NaN scores clamp to -1e30 first (``dsv4_topk.py:85``).
    """
    rows = scores.shape[0]
    ids = np.zeros((rows, k), dtype=np.int32)
    current = np.where(np.isnan(scores), -1e30, scores).astype(np.float32)
    current = current + bias.astype(np.float32)[None, :]
    for r in range(rows):
        row = current[r].copy()
        for slot in range(k):
            top = np.max(row)
            winners = np.flatnonzero(row == top)
            pick = int(winners[0])  # lowest id among ties
            ids[r, slot] = pick
            row[pick] = -np.inf
    return ids, current


def _weights_oracle(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    weights = np.take_along_axis(scores, ids, axis=1).astype(np.float32)
    total = weights.sum(axis=-1, keepdims=True)
    denom = np.where(total > 0.0, total, np.float32(1.0))
    return (weights * (np.float32(_SCALE) / denom)).astype(np.float32)


def _make_router(weight_bf: np.ndarray, bias: np.ndarray) -> Router:
    router = Router(_HIDDEN, num_experts=_EXPERTS, num_experts_per_tok=_TOPK)
    router.weight = F.cast(Tensor.from_dlpack(weight_bf), DType.bfloat16)
    router.bias = Tensor.from_dlpack(bias.astype(np.float32))
    return router


def test_sqrt_softplus_stable_formula() -> None:
    """The stable softplus matches the naive one on moderate logits."""
    rng = np.random.default_rng(12)
    logits = (rng.standard_normal((4, 128)) * 3.0).astype(np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(logits)).to_numpy()
    expected = np.sqrt(np.log1p(np.exp(logits))).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # Very negative logits: the naive form underflows to 0, the stable
    # form keeps the linear branch (matches the oracle's linear regime).
    extreme = np.array([[-100.0, -20.0, 0.0, 20.0, 100.0]], np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(extreme)).to_numpy()
    expected = _sqrt_softplus_oracle(extreme)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_stable_top_k_ties_and_nan() -> None:
    """Ties resolve to the lowest expert id; NaNs never win a slot."""
    scores = np.zeros((2, 8), dtype=np.float32)
    scores[0] = np.array([0.1, 0.3, 0.3, 0.2, 0.3, 0.0, 0.3, 0.3], np.float32)
    # Row 1: all NaN -- every slot clamps to -1e30, so the row yields six
    # DISTINCT ascending ids (the -inf picked-mask keeps them distinct).
    scores[1] = np.nan
    got = _stable_top_k(Tensor.from_dlpack(scores), 6, 8).to_numpy()
    # Row 0: four experts tie at 0.3 (ids 1, 2, 4, 6, 7 -- five of them);
    # each slot takes the tied max, lowest id first.
    np.testing.assert_array_equal(
        got[0], np.array([1, 2, 4, 6, 7, 3], np.int32)
    )
    np.testing.assert_array_equal(
        got[1], np.array([0, 1, 2, 3, 4, 5], np.int32)
    )


def test_router_production_shape_noaux_tc() -> None:
    """Full widths: selection uses the biased scores, weights the unbiased."""
    rng = np.random.default_rng(13)
    rows = 8
    weight = (rng.standard_normal((_EXPERTS, _HIDDEN)) * 0.02).astype(
        np.float32
    )
    bias = (rng.standard_normal(_EXPERTS) * 0.5).astype(np.float32)
    x = (rng.standard_normal((rows, _HIDDEN)) * 0.5).astype(np.float32)
    weight_bf = _to_bf16(weight)

    router = _make_router(weight_bf, bias)
    ids, weights = router(Tensor.from_dlpack(x), None)
    got_ids = ids.to_numpy()
    got_weights = weights.to_numpy()

    logits = x.astype(np.float32) @ weight_bf.astype(np.float32).T
    scores = _sqrt_softplus_oracle(logits)
    expected_ids, _ = _select_oracle(scores, bias, _TOPK)
    np.testing.assert_array_equal(got_ids, expected_ids)
    expected_weights = _weights_oracle(scores, expected_ids)
    np.testing.assert_allclose(
        got_weights, expected_weights, rtol=1e-5, atol=1e-6
    )


def test_router_bias_steers_selection_not_weights() -> None:
    """A huge bias on one expert forces its selection; weights unchanged."""
    rng = np.random.default_rng(14)
    weight = (rng.standard_normal((_EXPERTS, _HIDDEN)) * 0.02).astype(
        np.float32
    )
    bias = np.zeros(_EXPERTS, dtype=np.float32)
    bias[7] = 50.0
    x = (rng.standard_normal((3, _HIDDEN)) * 0.5).astype(np.float32)
    weight_bf = _to_bf16(weight)

    router = _make_router(weight_bf, bias)
    ids, weights = router(Tensor.from_dlpack(x), None)
    got_ids, got_weights = ids.to_numpy(), weights.to_numpy()

    logits = x.astype(np.float32) @ weight_bf.astype(np.float32).T
    scores = _sqrt_softplus_oracle(logits)
    expected_ids, _ = _select_oracle(scores, bias, _TOPK)
    np.testing.assert_array_equal(got_ids, expected_ids)
    # Expert 7 wins a slot on every row purely through the bias...
    assert (got_ids == 7).any(axis=1).all()
    # ...but its routing weight is its UNBIASED score, renormalized.
    expected_weights = _weights_oracle(scores, expected_ids)
    np.testing.assert_allclose(
        got_weights, expected_weights, rtol=1e-5, atol=1e-6
    )


def test_router_all_zero_row_ties_to_lowest_ids() -> None:
    """A zero input ties every expert: the six lowest ids win, weights even."""
    weight = np.full((_EXPERTS, _HIDDEN), 0.01, dtype=np.float32)
    bias = np.zeros(_EXPERTS, dtype=np.float32)
    x = np.zeros((2, _HIDDEN), dtype=np.float32)
    router = _make_router(_to_bf16(weight), bias)
    ids, weights = router(Tensor.from_dlpack(x), None)
    np.testing.assert_array_equal(
        ids.to_numpy(), np.tile(np.arange(_TOPK, dtype=np.int32), (2, 1))
    )
    # Equal scores renormalize to scaling / k each.
    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full((2, _TOPK), _SCALE / _TOPK, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_router_image_rows_swap_bias_vl() -> None:
    """Image rows select with ``bias_vl``; text rows keep ``bias``."""
    rng = np.random.default_rng(15)
    weight = (rng.standard_normal((_EXPERTS, _HIDDEN)) * 0.02).astype(
        np.float32
    )
    bias = np.zeros(_EXPERTS, dtype=np.float32)
    bias_vl = np.zeros(_EXPERTS, dtype=np.float32)
    bias_vl[11] = 40.0
    x = (rng.standard_normal((2, _HIDDEN)) * 0.5).astype(np.float32)
    weight_bf = _to_bf16(weight)

    router = Router(
        _HIDDEN,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOPK,
        with_bias_vl=True,
    )
    router.weight = F.cast(Tensor.from_dlpack(weight_bf), DType.bfloat16)
    router.bias = Tensor.from_dlpack(bias)
    router.bias_vl = Tensor.from_dlpack(bias_vl)
    mask = np.array([1, 0], dtype=np.uint8)

    ids, _ = router(Tensor.from_dlpack(x), Tensor.from_dlpack(mask))
    got_ids = ids.to_numpy()
    # The image row picks expert 11 first; the text row never does.
    assert got_ids[0, 0] == 11
    assert not (got_ids[1] == 11).any()


def test_router_rejects_unsupported_topk() -> None:
    import pytest

    with pytest.raises(ValueError):
        Router(_HIDDEN, num_experts=_EXPERTS, num_experts_per_tok=8)
