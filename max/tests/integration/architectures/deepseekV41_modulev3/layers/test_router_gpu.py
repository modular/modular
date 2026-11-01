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
"""GPU tests for the V4.1 router on main's ``moe_router_image_bias`` kernel.

Production shape: 5120-wide inputs against the 384-expert bf16 gate, top-6,
scaling 1.5. The oracle is an independent NumPy transcription of the
selection loop the composed router ran (HF ``Gate``, vLLM ``dsv4_topk``):
max per slot, lowest id among ties, NaN scores clamped below every real
score. Expert ids must match it exactly, ties and NaN rows included.
"""

from __future__ import annotations

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.router import (
    Router,
)
from max.tensor import Tensor

_HIDDEN = 5120
_EXPERTS = 384
_TOPK = 6
_SCALE = 1.5


def _gpu(t: np.ndarray, dtype: DType) -> Tensor:
    return Tensor.from_dlpack(t).to(Accelerator()).cast(dtype)


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


def _make_router(
    weight_bf: np.ndarray, bias: np.ndarray, bias_vl: np.ndarray | None = None
) -> Router:
    router = Router(
        _HIDDEN,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOPK,
        with_bias_vl=bias_vl is not None,
    )
    router.weight = _gpu(weight_bf, DType.bfloat16)
    router.bias = _gpu(bias.astype(np.float32), DType.float32)
    if bias_vl is not None:
        router.bias_vl = _gpu(bias_vl.astype(np.float32), DType.float32)
    return router


def _route(
    router: Router, x: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    ids, weights = router(
        _gpu(x, DType.float32),
        None if mask is None else _gpu(mask, DType.uint8),
    )
    return ids.to_numpy(), weights.to_numpy()


def _logits(router: Router, x: np.ndarray) -> np.ndarray:
    """The router's own gate GEMM, read back: float32 on a B200 is not
    bit-exact with NumPy's, so near-ties in random data would otherwise
    differ in the oracle's inputs, not in the kernel."""
    gate = _gpu(x, DType.float32) @ router.weight.cast(DType.float32).T
    return gate.to_numpy()


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
    got_ids, got_weights = _route(router, x)

    logits = _logits(router, x)
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
    got_ids, got_weights = _route(router, x)

    logits = _logits(router, x)
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
    got_ids, got_weights = _route(router, x)
    np.testing.assert_array_equal(
        got_ids, np.tile(np.arange(_TOPK, dtype=np.int32), (2, 1))
    )
    # Equal scores renormalize to scaling / k each.
    np.testing.assert_allclose(
        got_weights,
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

    router = _make_router(weight_bf, bias, bias_vl)
    mask = np.array([1, 0], dtype=np.uint8)

    got_ids, _ = _route(router, x, mask)
    # The image row picks expert 11 first; the text row never does.
    assert got_ids[0, 0] == 11
    assert not (got_ids[1] == 11).any()


def test_router_ties_and_nan_rows_match_the_stable_loop_exactly() -> None:
    """Controlled logits: heavy ties, an all-NaN row and image rows.

    The gate weight is a selector (expert ``e`` reads input column ``e``),
    so the logits are the bf16-exact input values and equal values tie
    exactly. A NaN input poisons every logit, so that row's scores are all
    NaN: ids must be the six lowest, and its weights stay NaN.
    """
    rng = np.random.default_rng(16)
    rows = 24
    weight = np.zeros((_EXPERTS, _HIDDEN), dtype=np.float32)
    weight[np.arange(_EXPERTS), np.arange(_EXPERTS)] = 1.0
    # Eight distinct logit levels over 384 experts: dozens of experts tie
    # at every level, in every lane of the kernel's warp layout.
    x = np.zeros((rows, _HIDDEN), dtype=np.float32)
    x[:, :_EXPERTS] = rng.integers(0, 8, (rows, _EXPERTS)).astype(np.float32)
    x[3, 5] = np.nan
    bias = np.zeros(_EXPERTS, dtype=np.float32)
    bias_vl = np.zeros(_EXPERTS, dtype=np.float32)
    bias_vl[100:140] = 0.5
    mask = (np.arange(rows) % 3 == 0).astype(np.uint8)

    router = _make_router(weight, bias, bias_vl)
    got_ids, got_weights = _route(router, x, mask)

    scores = _sqrt_softplus_oracle(x[:, :_EXPERTS])
    scores[3] = np.nan
    want_ids = np.zeros((rows, _TOPK), dtype=np.int32)
    for r in range(rows):
        row_bias = bias_vl if mask[r] else bias
        want_ids[r] = _select_oracle(scores[r : r + 1], row_bias, _TOPK)[0][0]
    np.testing.assert_array_equal(got_ids, want_ids)
    np.testing.assert_array_equal(got_ids[3], np.arange(_TOPK))
    assert np.isnan(got_weights[3]).all()
    live = np.arange(rows) != 3
    np.testing.assert_allclose(
        got_weights[live],
        _weights_oracle(scores, want_ids)[live],
        rtol=1e-5,
        atol=1e-6,
    )
