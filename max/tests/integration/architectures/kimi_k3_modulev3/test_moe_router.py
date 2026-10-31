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
"""Numerical verification for :func:`make_kimi_k3_moe_router`.

The reference below is adapted from V2's own trusted transcription of the
official reference (``kimi_k3/tests/test_moe_router.py``), copied here rather
than imported so a shared mistake between the layer and its reference could
not hide a bug.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, accelerator_architecture_name
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.deepseekV3_modulev3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_router import (
    KimiK3TopKRouter,
    make_kimi_k3_moe_router,
)
from max.tensor import Tensor, default_device, default_dtype

_TOKENS = 12
_HIDDEN = 32
_EXPERTS = 64
_TOP_K = 4
_SCALE = 2.5

# The production gate contraction depth and expert count. 7168 is a whole
# number of the fused GEMV's 512-element K tiles on gfx950.
_PROD_HIDDEN = 7168
_PROD_EXPERTS = 896
_PROD_TOP_K = 16

_MIXED_GEMV_OP = "mo.router.gate.mixed.gemv"
_ROUTER_MODULE = (
    "max.pipelines.architectures.kimi_k3_modulev3.layers.moe_router"
)

# Selection-boundary margin every case below must clear: far above the
# fp32-vs-float64 rounding that could reorder two nearly-tied experts, so
# exact top-k index equality is a property of the math, not of luck.
_SELECTION_MARGIN = 1e-2

# The same margin scaled to the production expert count: the 16th-vs-17th
# corrected-score gap of 896 near-iid values sits near 1 / (896 * density)
# ~ 3e-3, an order below the 64-expert bar, while the fp32-vs-float64
# score deviation stays ~2e-5 (measured). 1e-3 keeps ~50x headroom over the
# rounding noise.
_PROD_SELECTION_MARGIN = 1e-3


def _reference(
    hidden: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    bias: npt.NDArray[np.float32],
    num_expert_group: int,
    topk_group: int,
    tokens: int = _TOKENS,
    num_experts: int = _EXPERTS,
    top_k: int = _TOP_K,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], np.float64]:
    """Transcribes ``KimiMoEGate.forward``, ``modeling_kimi_linear.py:775-838``.

    Run in float64 so a disagreement is the gate's logic and not the
    reference's own rounding. Returns the selected indices, their scaled
    weights, and the smallest gap between the last selected and first
    unselected corrected score across tokens.
    """
    logits = hidden.astype(np.float64) @ weight.astype(np.float64).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    scores_for_choice = scores + bias.astype(np.float64)[None, :]
    if num_expert_group > 1 and num_expert_group > topk_group:
        grouped = scores_for_choice.reshape(tokens, num_expert_group, -1)
        group_scores = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
        group_idx = np.argsort(-group_scores, axis=-1)[:, :topk_group]
        keep = np.zeros_like(group_scores, dtype=bool)
        np.put_along_axis(keep, group_idx, True, axis=-1)
        keep = np.repeat(keep, num_experts // num_expert_group, axis=-1)
        tmp_scores = np.where(keep, scores_for_choice, -np.inf)
    else:
        tmp_scores = scores_for_choice
    topk_idx = np.argsort(-tmp_scores, axis=-1)[:, :top_k]
    topk_weight = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_weight /= topk_weight.sum(axis=-1, keepdims=True) + 1e-20
    ordered = np.sort(tmp_scores, axis=-1)[:, ::-1]
    boundary_gap = np.min(ordered[:, top_k - 1] - ordered[:, top_k])
    return topk_idx, topk_weight * _SCALE, boundary_gap


def _random_inputs(
    seed: int,
    tokens: int = _TOKENS,
    hidden: int = _HIDDEN,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((tokens, hidden)).astype(np.float32),
        rng.standard_normal((_EXPERTS, hidden)).astype(np.float32),
        rng.standard_normal(_EXPERTS).astype(np.float32),
    )


def _round_to_bfloat16(
    values: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Rounds fp32 values to bf16-representable ones, round-to-nearest-even.

    Feeding only bf16-representable values makes the device's fp32-to-bf16
    cast exact, so the reference consumes bit-identical inputs to the ones
    the router reads.
    """
    bits = values.view(np.uint32)
    rounded = (
        bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


@contextmanager
def _arch(architecture: str) -> Iterator[MagicMock]:
    """Answers the router's accelerator query as ``architecture`` would."""
    with patch(
        f"{_ROUTER_MODULE}.accelerator_architecture_name",
        return_value=architecture,
    ) as query:
        yield query


_HOST_DEVICE = DeviceRef.CPU()


def _trace_router(
    router: DeepseekV3TopKRouter,
    dtype: DType,
    hidden: int,
    device: DeviceRef = _HOST_DEVICE,
) -> str:
    """Traces the router's graph for one activation dtype.

    The fused path is device-gated, so tests that expect the op trace on
    the accelerator device and tests that expect it gated off trace on the
    host. The router must be constructed under the same device as the
    trace, so its externalized weights land on the graph's device.
    """
    staged = router.trace(TensorType(dtype, ["m", hidden], device=device))
    return str(staged)


def _production_router() -> KimiK3TopKRouter:
    """Builds the router at Kimi K3's production gate shape."""
    return make_kimi_k3_moe_router(
        hidden_dim=_PROD_HIDDEN,
        num_experts=_PROD_EXPERTS,
        num_experts_per_token=_PROD_TOP_K,
        routed_scaling_factor=1.0,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
    )


@pytest.mark.compile_on_gpu(
    "the mo.moe router kernels are staged as an undeclared custom op the interpreter can't bind"
)
@pytest.mark.parametrize(
    ("num_expert_group", "topk_group"),
    [(1, 1), (8, 4)],
    ids=["ungrouped", "group_limited"],
)
def test_matches_independent_reference(
    num_expert_group: int, topk_group: int
) -> None:
    """Published K3 runs ungrouped (``num_expert_group=1``); the
    group-limited case is checked too so a mask that silently selected
    everything would not pass by coincidence."""
    hidden_np, weight_np, bias_np = _random_inputs(0)
    expected_idx, expected_weight, boundary_gap = _reference(
        hidden_np, weight_np, bias_np, num_expert_group, topk_group
    )
    assert boundary_gap > _SELECTION_MARGIN, (
        "fixture's selection boundary is too close to a tie for exact "
        f"index equality: gap={boundary_gap}"
    )

    with default_device(Accelerator()):
        router = make_kimi_k3_moe_router(
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            moe_renormalize=True,
        )
        router.gate_score.weight = Tensor(weight_np, dtype=DType.float32)
        router.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)

        hidden = Tensor(hidden_np, dtype=DType.float32)
        idx, weight = router(hidden)
        idx_np = idx.cast(DType.int64).to_numpy()
        weight_np_out = weight.to_numpy().astype(np.float64)

    # Selection must match exactly: a biased-vs-unbiased gather mistake still
    # produces plausible-looking weights at the WRONG indices.
    np.testing.assert_array_equal(
        np.sort(idx_np, axis=-1), np.sort(expected_idx, axis=-1)
    )
    rel_error = np.max(np.abs(weight_np_out - expected_weight)) / max(
        np.max(np.abs(expected_weight)), 1e-12
    )
    # The shared `group_limited_router_kernel`'s GPU sigmoid/exp carries
    # about this much deviation from a float64 CPU reference -- DeepSeek
    # V2's own test of the same kernel (`test_moe_gate.py`) uses the same
    # `1e-3` rtol, not a tighter one, for this reason.
    assert rel_error < 1e-3, f"router weights diverge: rel_error={rel_error}"


@pytest.mark.compile_on_gpu(
    "the mo.moe router kernels are staged as an undeclared custom op the interpreter can't bind"
)
@pytest.mark.parametrize("tokens", [1, 12], ids=["decode_row", "rows"])
@pytest.mark.parametrize(
    ("num_expert_group", "topk_group"),
    [(1, 1), (8, 4)],
    ids=["ungrouped", "group_limited"],
)
def test_bf16_activations_match_independent_reference(
    tokens: int, num_expert_group: int, topk_group: int
) -> None:
    """bf16 activations match the reference at the production contraction
    depth, on gfx950 through the fused mixed GEMV.

    The inputs are signed with mixed magnitudes (rows scaled by powers of
    ten) and rounded to bf16-representable values, so the reference -- a
    float64 cast-then-matmul over the exact bf16 bits -- and the kernel
    widen identical values. The fused path needs the K-alignment this
    fixture's 7168 grants; ``_HIDDEN`` above is deliberately too small for
    it, which keeps this module's shrunk tests on the inherited path.
    """
    rng = np.random.default_rng(36)
    scales = 10.0 ** rng.integers(-2, 1, size=(tokens, 1))
    hidden_np = (rng.standard_normal((tokens, _PROD_HIDDEN)) * scales).astype(
        np.float32
    )
    weight_np = rng.standard_normal((_EXPERTS, _PROD_HIDDEN)).astype(np.float32)
    bias_np = rng.standard_normal(_EXPERTS).astype(np.float32)
    hidden_bf16 = _round_to_bfloat16(hidden_np)

    expected_idx, expected_weight, boundary_gap = _reference(
        hidden_bf16,
        weight_np,
        bias_np,
        num_expert_group,
        topk_group,
        tokens,
    )
    assert boundary_gap > _SELECTION_MARGIN, (
        "fixture's selection boundary is too close to a tie for exact "
        f"index equality: gap={boundary_gap}"
    )

    with default_device(Accelerator()):
        router = make_kimi_k3_moe_router(
            hidden_dim=_PROD_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            moe_renormalize=True,
        )
        router.gate_score.weight = Tensor(weight_np, dtype=DType.float32)
        router.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)

        hidden = Tensor(hidden_bf16, dtype=DType.float32).cast(DType.bfloat16)
        idx, weight = router(hidden)
        idx_np = idx.cast(DType.int64).to_numpy()
        weight_np_out = weight.to_numpy().astype(np.float64)

    np.testing.assert_array_equal(
        np.sort(idx_np, axis=-1), np.sort(expected_idx, axis=-1)
    )
    rel_error = np.max(np.abs(weight_np_out - expected_weight)) / max(
        np.max(np.abs(expected_weight)), 1e-12
    )
    assert rel_error < 1e-3, f"router weights diverge: rel_error={rel_error}"


@pytest.mark.compile_on_gpu(
    "the mo.moe router kernels are staged as an undeclared custom op the interpreter can't bind"
)
@pytest.mark.parametrize(
    "tokens",
    [1, 4, 12],
    ids=["m1_streaming", "m4_wide_tm2", "m12_wide_tm4"],
)
def test_bf16_activations_match_independent_reference_at_896_experts(
    tokens: int,
) -> None:
    """bf16 activations match the reference at the production expert count
    and contraction depth, through the ``[896, 7168]`` cold-weight band
    table.

    The N=896 tests above only trace; this is the executed check of the
    table's bands. Tokens 1, 4 and 12 cross the three bands a decode batch
    reaches: the streaming ``m == 1`` row and the two wide ``act_wide_k``
    rows (``m = 2..6`` and ``m = 7..12``).
    """
    rng = np.random.default_rng(39)
    scales = 10.0 ** rng.integers(-2, 1, size=(tokens, 1))
    hidden_np = (rng.standard_normal((tokens, _PROD_HIDDEN)) * scales).astype(
        np.float32
    )
    weight_np = rng.standard_normal((_PROD_EXPERTS, _PROD_HIDDEN)).astype(
        np.float32
    )
    bias_np = rng.standard_normal(_PROD_EXPERTS).astype(np.float32)
    hidden_bf16 = _round_to_bfloat16(hidden_np)

    expected_idx, expected_weight, boundary_gap = _reference(
        hidden_bf16,
        weight_np,
        bias_np,
        1,
        1,
        tokens,
        num_experts=_PROD_EXPERTS,
        top_k=_PROD_TOP_K,
    )
    assert boundary_gap > _PROD_SELECTION_MARGIN, (
        "fixture's selection boundary is too close to a tie for exact "
        f"index equality: gap={boundary_gap}"
    )

    with default_device(Accelerator()):
        router = make_kimi_k3_moe_router(
            hidden_dim=_PROD_HIDDEN,
            num_experts=_PROD_EXPERTS,
            num_experts_per_token=_PROD_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        router.gate_score.weight = Tensor(weight_np, dtype=DType.float32)
        router.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)

        hidden = Tensor(hidden_bf16, dtype=DType.float32).cast(DType.bfloat16)
        idx, weight = router(hidden)
        idx_np = idx.cast(DType.int64).to_numpy()
        weight_np_out = weight.to_numpy().astype(np.float64)

    np.testing.assert_array_equal(
        np.sort(idx_np, axis=-1), np.sort(expected_idx, axis=-1)
    )
    rel_error = np.max(np.abs(weight_np_out - expected_weight)) / max(
        np.max(np.abs(expected_weight)), 1e-12
    )
    assert rel_error < 1e-3, f"router weights diverge: rel_error={rel_error}"


def test_gate_weight_is_forced_to_fp32_under_a_different_ambient_dtype() -> (
    None
):
    """The gate matmul must run in fp32 even when constructed under a
    non-fp32 ambient default dtype -- the whole reason this file wraps
    ``DeepseekV3TopKRouter`` in a ``default_dtype(DType.float32)`` context."""
    with default_device(Accelerator()), default_dtype(DType.bfloat16):
        router = make_kimi_k3_moe_router(
            hidden_dim=_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        assert router.gate_score.weight.dtype == DType.float32


def test_router_graph_takes_the_fused_gemv_on_gfx950() -> None:
    """The K3 router graph for an AMD gfx950 device with bf16 activations
    contains the fused mixed GEMV, at the production gate shape."""
    with _arch("gfx950"), default_device(Accelerator()):
        text = _trace_router(
            _production_router(),
            DType.bfloat16,
            _PROD_HIDDEN,
            device=DeviceRef.GPU(),
        )
    assert _MIXED_GEMV_OP in text


@pytest.mark.parametrize(
    ("dtype", "architecture", "on_gpu"),
    [
        (DType.float32, "gfx950", True),
        (DType.bfloat16, "sm_100a", True),
        (DType.bfloat16, "gfx950", False),
    ],
    ids=["fp32_activations", "off_gfx950", "cpu_device_graph"],
)
def test_router_graph_gates_off_the_fused_gemv(
    dtype: DType, architecture: str, on_gpu: bool
) -> None:
    """The fused GEMV is opt-in: fp32 activations on gfx950, bf16
    activations anywhere else, and a bf16 CPU-device graph even on gfx950
    (the op is registered GPU-only) keep the inherited cast + matmul
    path. The dtype and architecture cases trace on the GPU device, so the
    device check alone cannot gate them off."""
    with (
        _arch(architecture),
        default_device(Accelerator() if on_gpu else CPU()),
    ):
        text = _trace_router(
            _production_router(),
            dtype,
            _PROD_HIDDEN,
            device=DeviceRef.GPU() if on_gpu else _HOST_DEVICE,
        )
    assert _MIXED_GEMV_OP not in text


def test_router_graph_skips_the_arch_query_without_an_accelerator() -> None:
    """With no accelerator the architecture query raises, so the router
    must not reach it: the graph keeps the inherited path and builds."""
    with (
        patch(f"{_ROUTER_MODULE}.accelerator_count", return_value=0),
        patch(
            f"{_ROUTER_MODULE}.accelerator_architecture_name",
            side_effect=RuntimeError("no accelerator"),
        ),
        default_device(Accelerator()),
    ):
        text = _trace_router(
            _production_router(),
            DType.bfloat16,
            _PROD_HIDDEN,
            device=DeviceRef.GPU(),
        )
    assert _MIXED_GEMV_OP not in text


def test_deepseek_v3_router_graph_keeps_the_matmul() -> None:
    """The base router this one subclasses is untouched: on gfx950, with a
    bf16 gate under a bf16 ambient dtype, its graph holds no fused GEMV."""
    # On the GPU device: building a bf16 gate weight eagerly is a device
    # op, and the CPU interpreter refuses bf16 casts.
    with (
        _arch("gfx950"),
        default_device(Accelerator()),
        default_dtype(DType.bfloat16),
    ):
        router = DeepseekV3TopKRouter(
            hidden_dim=_PROD_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=8,
            topk_group=4,
            norm_topk_prob=True,
            correction_bias_dtype=DType.bfloat16,
        )
        text = _trace_router(
            router, DType.bfloat16, _PROD_HIDDEN, device=DeviceRef.GPU()
        )
    assert _MIXED_GEMV_OP not in text


@pytest.mark.skipif(
    accelerator_architecture_name() != "gfx950",
    reason="asserts the real accelerator string the production MI355X path reads",
)
def test_router_graph_takes_the_fused_gemv_on_this_gfx950_host() -> None:
    """Without any patching, a gfx950 host's own router graph takes the
    fused GEMV -- the patched tests above cannot catch a query that never
    matches the real architecture name."""
    with default_device(Accelerator()):
        text = _trace_router(
            _production_router(),
            DType.bfloat16,
            _PROD_HIDDEN,
            device=DeviceRef.GPU(),
        )
    assert _MIXED_GEMV_OP in text


def test_router_raises_on_a_gate_weight_the_fused_gemv_cannot_bind() -> None:
    """On the gfx950 bf16 path, a gate weight that is not fp32 raises at
    call time instead of silently taking the inherited path."""
    # On the GPU device: the bf16 gate weight this feeds the wrapper is
    # built by an eager cast, which the CPU interpreter refuses.
    hidden_np, weight_np, bias_np = _random_inputs(1, hidden=_PROD_HIDDEN)
    with _arch("gfx950"), default_device(Accelerator()):
        router = make_kimi_k3_moe_router(
            hidden_dim=_PROD_HIDDEN,
            num_experts=_EXPERTS,
            num_experts_per_token=_TOP_K,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        router.gate_score.weight = Tensor(weight_np, dtype=DType.float32).cast(
            DType.bfloat16
        )
        router.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)
        hidden = Tensor(hidden_np, dtype=DType.float32).cast(DType.bfloat16)
        with pytest.raises(
            ValueError, match="gate_weight to have dtype float32"
        ):
            router(hidden)
