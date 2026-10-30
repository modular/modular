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
"""Numerical verification for the fused attention-residual mixture.

Verifies ``kimi_k3_attn_res_mix_norm`` -- the ``attn_res_mix_norm`` custom op
through the ModuleV3 wrapper -- against an independent fp64 numpy reference of
the unfused chain it replaces (stack, score, softmax, mix, round to the
storage dtype, RMSNorm with cast-then-multiply), against the device's own
unfused chain (the ``attn_res_mix`` op plus ``kimi_rms_norm``), and against
the compiled graph's kernel names: the fused site must not launch the
candidate-stack concat (``concat_fused``), the unfused ``attn_res_mix``
kernels or the rowwise norm.
"""

from __future__ import annotations

import re
from typing import cast

import numpy as np
import pytest
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.pipelines.architectures.kimi_k3_modulev3.layers.attn_residual import (
    kimi_k3_attn_res_mix_norm,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.norm import (
    kimi_rms_norm,
)
from max.tensor import Tensor, default_device
from max.tensor.realization_context import ensure_context

_MIX_EPS = 1e-5
_NORM_EPS = 1e-5
# A hidden width that is not a multiple of the kernels' fixed-width slices
# (so their masked tails run), and the production one.
_HIDDEN_RAGGED = 300
_PRODUCTION_HIDDEN = 7168
# Past the fused kernel's one-CTA-per-token coverage (1024 x 8), where the
# wrapper falls back to the unfused chain.
_OVERSIZED_HIDDEN = 8200


def _round_to_storage_dtype(x: np.ndarray, dtype: DType) -> np.ndarray:
    """Rounds fp64 values the way a store to ``dtype`` does.

    bf16 has no numpy dtype, so its round-to-nearest-even runs on the float32
    bit pattern; f32 rounds through numpy itself.
    """
    if dtype == DType.float32:
        return x.astype(np.float32).astype(np.float64)
    bits = x.astype(np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32).astype(np.float64)


def _numpy_unfused_chain(
    candidates: np.ndarray,
    proj_weight: np.ndarray,
    res_norm_weight: np.ndarray,
    out_norm_weight: np.ndarray,
    eps: float,
    norm_eps: float,
    dtype: DType,
) -> np.ndarray:
    """Independent fp64 reference of the unfused chain: score each candidate
    against ``res_norm_weight * proj_weight`` after RMS-normalizing it,
    softmax over the candidate axis, weighted sum over the RAW candidates
    (not the normalized ones -- the specific thing V2's own docstring warns a
    port gets backwards), round to the storage dtype, then RMS-normalize
    with cast-then-multiply and the stored ``out_norm_weight``.

    Every input is the storage-dtype-rounded value the op sees, so the
    reference's fp64 arithmetic starts from the same rounded inputs rather
    than from the pre-rounding float32 originals.
    """
    c64 = candidates.astype(np.float64)  # [T, C, H]
    rms = np.sqrt(np.mean(c64**2, axis=-1, keepdims=True) + eps)
    scored = (c64 / rms) * res_norm_weight.astype(np.float64)
    scores = scored @ proj_weight.astype(np.float64).reshape(-1)  # [T, C]
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    mixed = np.sum(c64 * weights[..., None], axis=1)
    mixed = _round_to_storage_dtype(mixed, dtype)
    inv_rms = 1.0 / np.sqrt(
        np.mean(mixed**2, axis=-1, keepdims=True) + norm_eps
    )
    normed = _round_to_storage_dtype(mixed * inv_rms, dtype)
    return _round_to_storage_dtype(
        normed * out_norm_weight.astype(np.float64), dtype
    )


def _assert_matches_reference(
    out_np: np.ndarray, expected: np.ndarray, dtype: DType, context: str
) -> None:
    """Per-element check against the fp64 chain reference. The reference is
    fp64 while the op accumulates in fp32, reassociates the sum of squares
    and approximates rsqrt, so the check needs a small tolerance -- nowhere
    near enough to hide a wrong mixture, which shifts whole elements.
    """
    rel, abs_floor = (
        (2.0**-6, 1.0e-3) if dtype == DType.bfloat16 else (1.0e-4, 1.0e-5)
    )
    over = np.abs(out_np - expected) - np.maximum(
        rel * np.abs(expected), abs_floor
    )
    worst = over.max()
    assert worst <= 0, (
        f"{context}: output outside the fp64-chain tolerance"
        f" (rel={rel}, abs_floor={abs_floor}), worst overshoot"
        f" {worst} at element {int(np.argmax(over))}"
    )


def _assert_within_storage_ulps(
    out_np: np.ndarray,
    expected: np.ndarray,
    dtype: DType,
    ulps: float,
    context: str,
) -> None:
    """Per-element check in storage-dtype ulps against another device output.

    The bound covers pure rounding divergence -- a reassociated sum of
    squares, a product one kernel contracts and the other rounds exactly;
    a flip lands one ulp away, and a value sitting on a binade boundary
    can measure as two because the ulp itself halves across it.
    """
    fraction_bits = 7 if dtype == DType.bfloat16 else 23
    magnitude = np.maximum(np.abs(expected), 2.0**-126)
    ulp = 2.0 ** (np.floor(np.log2(magnitude)) - fraction_bits)
    distance = np.abs(out_np - expected) / ulp
    worst = distance.max()
    assert worst <= ulps, (
        f"{context}: {worst:.2f} storage-dtype ulps from the"
        f" unfused chain (bound {ulps}) at element"
        f" {int(np.unravel_index(np.argmax(distance), distance.shape)[0])}"
    )


def _to_storage_tensor(arr: np.ndarray, dtype: DType) -> Tensor:
    t = Tensor(arr, dtype=DType.float32)
    return t.cast(dtype) if dtype == DType.bfloat16 else t


def _random_inputs(
    rng: np.random.Generator, tokens: int, num_candidates: int, hidden: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draws the op's four operands; the draw order fixes what a seed
    produces, so keep it the same in every test that uses this."""
    candidates_np = rng.standard_normal(
        (tokens, num_candidates, hidden)
    ).astype(np.float32)
    proj_np = rng.standard_normal((1, hidden)).astype(np.float32)
    res_norm_np = rng.standard_normal(hidden).astype(np.float32)
    out_norm_np = rng.standard_normal(hidden).astype(np.float32)
    return candidates_np, proj_np, res_norm_np, out_norm_np


def _unfused_attn_res_mix(
    candidates: Tensor,
    proj_weight: Tensor,
    norm_weight: Tensor,
    *,
    eps: float,
    out_type: TensorType,
) -> TensorValue:
    """The raw ``attn_res_mix`` op call, kept here because no production
    wrapper for it remains: it lets the fused op be compared against the
    device's own unfused chain rather than a re-implementation of it.
    """
    device = TensorValue(candidates).device
    local_out_type = TensorType(out_type.dtype, out_type.shape, device=device)
    return cast(
        TensorValue,
        ops.custom(
            "attn_res_mix",
            device=device,
            values=[
                TensorValue(candidates),
                TensorValue(proj_weight),
                TensorValue(norm_weight),
            ],
            out_types=[local_out_type],
            parameters={"eps": eps},
        )[0],
    )


_unfused_attn_res_mix_op = F.functional(_unfused_attn_res_mix)


def _unfused_mix_and_norm(
    prefix_sum: Tensor,
    block_residual: list[Tensor],
    proj_weight: Tensor,
    res_norm_weight: Tensor,
    out_norm_weight: Tensor,
) -> Tensor:
    """The unfused chain: candidate stack, ``attn_res_mix``, ``kimi_rms_norm``."""
    out_dtype = prefix_sum.dtype
    candidates = F.cast(
        F.stack(block_residual + [prefix_sum], axis=1), out_dtype
    )
    out_type = TensorType(out_dtype, prefix_sum.shape, device=prefix_sum.device)
    mixed = _unfused_attn_res_mix_op(
        candidates,
        F.cast(proj_weight, out_dtype),
        F.cast(res_norm_weight, out_dtype),
        eps=_MIX_EPS,
        out_type=out_type,
    )
    return kimi_rms_norm(mixed, F.cast(out_norm_weight, out_dtype), _NORM_EPS)


@pytest.mark.compile_on_gpu("attn_res_mix_norm has no custom.declare")
@pytest.mark.parametrize(
    "dtype,hidden,num_candidates",
    [
        *[(DType.bfloat16, _HIDDEN_RAGGED, count) for count in (1, 2, 9)],
        (DType.bfloat16, _PRODUCTION_HIDDEN, 1),
        (DType.bfloat16, _PRODUCTION_HIDDEN, 9),
        (DType.float32, _HIDDEN_RAGGED, 1),
        (DType.float32, _HIDDEN_RAGGED, 9),
        (DType.float32, _PRODUCTION_HIDDEN, 9),
        (DType.bfloat16, _OVERSIZED_HIDDEN, 3),
    ],
)
def test_fused_matches_independent_reference(
    dtype: DType, hidden: int, num_candidates: int
) -> None:
    """The fused op against an fp64 model of the unfused chain, over the
    candidate-count ladder (count 1 is the empty-stack case), a hidden width
    that is not a multiple of the kernels' slices, the production width and
    one past the fused kernel's coverage (unfused fallback, bf16 only: its
    fp32 split kernels miss the fp32 tolerance), in both storage dtypes,
    on signed data."""
    rng = np.random.default_rng(42)
    tokens = 2
    candidates_np, proj_np, res_norm_np, out_norm_np = _random_inputs(
        rng, tokens, num_candidates, hidden
    )

    expected = _numpy_unfused_chain(
        _round_to_storage_dtype(candidates_np.astype(np.float64), dtype),
        _round_to_storage_dtype(proj_np.astype(np.float64), dtype),
        _round_to_storage_dtype(res_norm_np.astype(np.float64), dtype),
        _round_to_storage_dtype(out_norm_np.astype(np.float64), dtype),
        _MIX_EPS,
        _NORM_EPS,
        dtype,
    )

    with default_device(Accelerator()):
        prefix_sum = _to_storage_tensor(candidates_np[:, -1, :], dtype)
        stack = [
            _to_storage_tensor(candidates_np[:, i, :], dtype)
            for i in range(num_candidates - 1)
        ]
        out = kimi_k3_attn_res_mix_norm(
            prefix_sum,
            stack,
            _to_storage_tensor(proj_np, dtype),
            _to_storage_tensor(res_norm_np, dtype),
            _to_storage_tensor(out_norm_np, dtype),
            _MIX_EPS,
            _NORM_EPS,
        )
        out_np = out.cast(DType.float32).to_numpy().astype(np.float64)

    assert out.dtype == dtype
    assert list(out.shape) == [tokens, hidden]
    _assert_matches_reference(
        out_np,
        expected,
        dtype,
        f"fused mix+norm (dtype={dtype}, hidden={hidden}, C={num_candidates})",
    )


@pytest.mark.compile_on_gpu("attn_res_mix_norm has no custom.declare")
@pytest.mark.parametrize("dtype", [DType.bfloat16, DType.float32])
def test_fused_matches_unfused_chain_on_gpu(dtype: DType) -> None:
    """The fused op against the device's own unfused chain on identical
    device inputs, at the production hidden width where both take the split
    path. The mix is not bit-identical -- the old split kernel's codegen
    contracts one product into its accumulation while the fused kernel
    rounds every product exactly -- and the two norms reassociate the sum
    of squares differently, so the check is an ulp distance, not a bit
    compare."""
    rng = np.random.default_rng(7)
    hidden, num_candidates, tokens = _PRODUCTION_HIDDEN, 5, 2
    candidates_np, proj_np, res_norm_np, out_norm_np = _random_inputs(
        rng, tokens, num_candidates, hidden
    )

    with default_device(Accelerator()):
        prefix_sum = _to_storage_tensor(candidates_np[:, -1, :], dtype)
        stack = [
            _to_storage_tensor(candidates_np[:, i, :], dtype)
            for i in range(num_candidates - 1)
        ]
        proj = _to_storage_tensor(proj_np, dtype)
        res_norm = _to_storage_tensor(res_norm_np, dtype)
        out_norm = _to_storage_tensor(out_norm_np, dtype)

        fused = kimi_k3_attn_res_mix_norm(
            prefix_sum, stack, proj, res_norm, out_norm, _MIX_EPS, _NORM_EPS
        )
        unfused = _unfused_mix_and_norm(
            prefix_sum, stack, proj, res_norm, out_norm
        )

        fused_np = fused.cast(DType.float32).to_numpy().astype(np.float64)
        unfused_np = unfused.cast(DType.float32).to_numpy().astype(np.float64)

    _assert_within_storage_ulps(
        fused_np,
        unfused_np,
        dtype,
        2.0 if dtype == DType.bfloat16 else 16.0,
        f"fused mix+norm vs unfused chain (dtype={dtype})",
    )


@pytest.mark.compile_on_gpu("attn_res_mix_norm has no custom.declare")
def test_fused_graph_runs_no_unfused_mix_kernels() -> None:
    """The fused site lowers to the fused pair only: the compiled graph
    must not contain the candidate-stack concat, the unfused
    ``attn_res_mix`` kernels or the rowwise norm."""
    hidden, num_candidates, tokens = _HIDDEN_RAGGED, 4, 2
    rng = np.random.default_rng(11)
    candidates_np, proj_np, res_norm_np, out_norm_np = _random_inputs(
        rng, tokens, num_candidates, hidden
    )

    candidate_type = TensorType(
        DType.float32, [tokens, hidden], device=DeviceRef.GPU()
    )
    proj_type = TensorType(DType.float32, [1, hidden], device=DeviceRef.GPU())
    weight_type = TensorType(DType.float32, [hidden], device=DeviceRef.GPU())
    input_types = [
        *[candidate_type] * num_candidates,
        proj_type,
        weight_type,
        weight_type,
    ]
    with Graph("fused_attn_res_mix_norm", input_types=input_types) as graph:
        # The wrapper runs on eager Tensors, so wrap the graph's values in
        # a realization context first: the tensors stay symbolic, and every
        # op the wrapper adds lands in this graph.
        with ensure_context():
            values = [Tensor.from_graph_value(v.tensor) for v in graph.inputs]
            # Operand order: block residuals first, the running prefix last,
            # then the three weight operands.
            *block_residuals, prefix, proj, res_norm, out_norm = values
            out = kimi_k3_attn_res_mix_norm(
                prefix,
                block_residuals,
                proj,
                res_norm,
                out_norm,
                _MIX_EPS,
                _NORM_EPS,
            )
        graph.output(out)

    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    summaries = model.kernel_summaries
    assert any("attn_res_mix_norm" in s for s in summaries), summaries
    banned = re.compile(
        r"concat_fused|attn_res_mix_gpu|attn_res_score_partials"
        r"|attn_res_mix_from_partials|rowwise"
    )
    assert not any(banned.search(s) for s in summaries), summaries

    arrays = [candidates_np[:, i, :] for i in range(num_candidates)] + [
        proj_np,
        res_norm_np,
        out_norm_np,
    ]
    device_inputs = [
        Buffer.from_numpy(arr).to(model.input_devices[i])
        for i, arr in enumerate(arrays)
    ]
    (result,) = model.execute(*device_inputs)
    out_np = result.to_numpy()
    assert out_np.shape == (tokens, hidden)
    assert np.isfinite(out_np).all()
