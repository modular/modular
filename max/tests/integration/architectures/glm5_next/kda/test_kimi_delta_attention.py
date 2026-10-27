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
"""One KDA layer, tensor-level, against the pinned torch reference.

Scoped to a single layer deliberately: the checkpoint has 34 of them and an
end-to-end logit check cannot attribute a mismatch to any one. Three of the
near-misses this layer is exposed to -- a permuted QKV conv concat, a gated norm
that normalizes in the wrong order, and a forget gate collapsed from per channel
to per head -- produce a layer that runs and answers slightly badly, so each is
checked as a *differential*: the layer must match the reference and must differ
materially from the plausible-but-wrong variant.

Dimensions are the real model's -- 64 heads of 128, ``hidden_size`` 4096, conv
width 24576 -- because the recurrence kernel is compiled per head-dimension pair
and a narrowed fixture would exercise code the model never runs. The head count
in particular: nothing else in the tree runs the KDA recurrence at
``num_heads == 64``.

Everything is float32 here, weights included, so the comparison is not blunted
by bfloat16 rounding. The real model runs bfloat16 projections into the same
float32 recurrence and float32 gated norm. The one exception is prefill on
gfx950, where the layer rounds q/k/v and the recurrence output through bf16
for the chunk kernel.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
)
from max.nn.legacy.layer import Module
from max.pipelines.architectures.glm5_next.layers import (
    KdaReplayInputs,
    KimiDeltaAttention,
)
from torch_reference import (
    GatedNorm,
    KdaWeights,
    as_dtype,
    gate_then_norm,
    kda_layer,
)

_NUMPY_DTYPE = {torch.float32: np.float32, torch.float64: np.float64}

pytestmark = pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")

HIDDEN = 4096
HEADS = 64
HEAD_DIM = 128
KERNEL = 4
QKV_DIM = HEADS * HEAD_DIM
CONV_DIM = 3 * QKV_DIM
RMS_NORM_EPS = 1e-5
LOWER_BOUND = -5.0

MAX_SLOTS = 4
SLOTS = (1, 3)
"""Pool slots the batch occupies; 0 and 2 must come back untouched."""

PREFILL_LENGTHS = (5, 3)

# The layer's output is a sum of 8192 same-magnitude terms, so individual
# elements cancel to near zero and an elementwise *relative* tolerance is
# meaningless on them -- the worst single element sits 55x off while the tensor
# as a whole agrees to a part in a thousand. The bound is therefore
# RMSE-relative, the metric `test_kda_chunk_fwd.mojo` and `test_kda_ops.py`
# already use for this recurrence.
#
# Where the budget comes from: MAX's float32 matmul on this GPU is not IEEE
# float32. `test_layer_matches_reference` evaluates the same formula in float64
# and finds the float32 *torch* reference 5.2e-7 away from it, while MAX sits
# three orders of magnitude further out -- the signature of tensor-core matmul
# rounding its inputs to a 10-bit mantissa. Four chained projections
# (q/k/v, f_a->f_b, g_a->g_b, o_proj) each contribute up to one such rounding,
# so the budget is a small multiple of that unit rather than a round number.
TENSOR_CORE_ULP = 2.0**-11
LAYER_REL_RMS = 16 * TENSOR_CORE_ULP

MULTI_CHUNK_PREFILL_LENGTHS = (130, 67)
"""Three and two of the gfx950 chunk kernel's 64-token chunks, with tails."""
CONTINUATION_DECODE_STEPS = 4

CARRY_GATE_ARGUMENT = -6.0
"""Mean of ``exp(A_log) * (g + dt_bias)`` in the continuation fixture.

The bounded gate is ``LOWER_BOUND * sigmoid`` of it, about -0.012 per token, so
a state keeps about a fifth of itself across the 130-token prefill. The shared
fixture centers it at 0, a decay of about -2.5 per token that erases the state
within a few tokens and so hides any error the prefill writes into it.
"""
CARRY_BETA_LOGIT = -5.0
"""Mean of beta's pre-sigmoid logit in the continuation fixture.

``b_proj`` has no bias, so the fixture's hidden states carry a constant 1 in
feature 0 and ``b_proj`` weights that feature by this value.
"""

REFERENCE_REL_RMS = 1e-5
"""How close the float32 reference must be to its own float64 evaluation.

Measured 5.2e-7. This is what says the transcription is arithmetically sound,
so a gap between MAX and the reference is MAX's matmul precision and not a
disagreement about the formula.
"""

WRONG_VARIANT_MIN_REL_DIFF = 0.05
"""How far a wrong-order variant must sit from the reference to be a real check.

A differential check proves nothing if the two arms agree anyway. Each variant
below is asserted to move the output by at least this much relative RMS, six
times :data:`LAYER_REL_RMS`. The measured drifts are far larger -- 1.21 to 1.30
for a permuted conv concat, 0.454 for a gate-then-normalise, 0.137 for a
per-head decay -- and the tightest of those is the one that matters most, since
it is the whole difference between KDA and Gated DeltaNet.
"""


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    """RMS of ``a - b`` over RMS of ``a``, in float64."""
    a64, b64 = a.double(), b.double()
    return float(
        torch.sqrt(((a64 - b64) ** 2).mean())
        / (torch.sqrt((a64**2).mean()) + 1e-12)
    )


class _KdaGraph(Module):
    """One :class:`KimiDeltaAttention` behind a positional call signature."""

    captured: list[KdaReplayInputs] | None = None
    """Set to a list to have the layer record its state-kernel inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.kda = KimiDeltaAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            conv_kernel_size=KERNEL,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            rms_norm_eps=RMS_NORM_EPS,
            lower_bound=LOWER_BOUND,
        )

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        recurrent_pool: BufferValue,
        row_id: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # One row id for both leaves: these pools hold a single layer, so a
        # request's row in each is its block, and the two block spaces
        # coincide. A real cache draws the two leaves' blocks separately.
        return self.kda(
            x,
            conv_pool,
            row_id,
            recurrent_pool,
            row_id,
            input_row_offsets,
            replay_capture=self.captured,
        )


def _input_types() -> list[TensorType | BufferType]:
    """Graph inputs, with symbolic token and batch extents.

    Symbolic so one compiled graph serves both the prefill and the one-token
    decode step; the ops read their extents from the tensors at run time.
    """
    gpu = DeviceRef.GPU()
    types: list[TensorType | BufferType] = [
        TensorType(DType.float32, ["total_tokens", HIDDEN], device=gpu),
        BufferType(
            DType.float32, [MAX_SLOTS, CONV_DIM, KERNEL - 1], device=gpu
        ),
        BufferType(
            DType.float32,
            [MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM],
            device=gpu,
        ),
        TensorType(DType.int32, ["batch"], device=gpu),
        TensorType(DType.uint32, ["batch_plus_one"], device=gpu),
    ]
    return types


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator(0)])


@pytest.fixture(scope="module")
def weights() -> dict[str, torch.Tensor]:
    """One layer's weights, with every per-channel parameter given a spread.

    ``dt_bias`` in particular varies within a head: it is the only thing that
    makes a per-channel forget gate distinguishable from a per-head one on
    random inputs, so it must not be constant.
    """
    generator = torch.Generator().manual_seed(20260826)

    def normal(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    return {
        "q_proj.weight": normal(QKV_DIM, HIDDEN, std=0.02),
        "k_proj.weight": normal(QKV_DIM, HIDDEN, std=0.02),
        "v_proj.weight": normal(QKV_DIM, HIDDEN, std=0.02),
        "conv1d.weight": normal(CONV_DIM, 1, KERNEL, std=0.5),
        "f_a_proj.weight": normal(HEAD_DIM, HIDDEN, std=0.02),
        "f_b_proj.weight": normal(QKV_DIM, HEAD_DIM, std=0.1),
        "dt_bias": normal(QKV_DIM, std=0.5),
        "A_log": normal(HEADS, std=0.5),
        "b_proj.weight": normal(HEADS, HIDDEN, std=0.02),
        "g_a_proj.weight": normal(HEAD_DIM, HIDDEN, std=0.02),
        "g_b_proj.weight": normal(QKV_DIM, HEAD_DIM, std=0.1),
        "o_norm.weight": 1.0 + normal(HEAD_DIM, std=0.05),
        "o_proj.weight": normal(HIDDEN, QKV_DIM, std=0.02),
    }


def _reference_weights(weights: dict[str, torch.Tensor]) -> KdaWeights:
    return KdaWeights(
        q_proj=weights["q_proj.weight"],
        k_proj=weights["k_proj.weight"],
        v_proj=weights["v_proj.weight"],
        conv1d=weights["conv1d.weight"],
        f_a_proj=weights["f_a_proj.weight"],
        f_b_proj=weights["f_b_proj.weight"],
        dt_bias=weights["dt_bias"],
        A_log=weights["A_log"],
        b_proj=weights["b_proj.weight"],
        g_a_proj=weights["g_a_proj.weight"],
        g_b_proj=weights["g_b_proj.weight"],
        o_norm=weights["o_norm.weight"],
        o_proj=weights["o_proj.weight"],
    )


@pytest.fixture(scope="module")
def reference_weights(weights: dict[str, torch.Tensor]) -> KdaWeights:
    return _reference_weights(weights)


@pytest.fixture(scope="module")
def carry_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """:func:`weights` with slow forget gates and low beta, so state carries.

    ``f_b_proj`` shrinks so the raw gate spreads by about 0.4 around
    ``dt_bias``, and ``A_log`` narrows so ``exp(A_log)`` stays near 1: the
    gate's sigmoid argument then sits near :data:`CARRY_GATE_ARGUMENT` in every
    head. ``dt_bias`` keeps its per-channel spread.
    """
    generator = torch.Generator().manual_seed(20261022)

    def normal(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    b_proj = normal(HEADS, HIDDEN, std=0.005)
    b_proj[:, 0] = CARRY_BETA_LOGIT
    return weights | {
        "f_b_proj.weight": normal(QKV_DIM, HEAD_DIM, std=0.03),
        "dt_bias": CARRY_GATE_ARGUMENT + normal(QKV_DIM, std=0.5),
        "A_log": normal(HEADS, std=0.1),
        "b_proj.weight": b_proj,
    }


def _carry_hidden_states(
    generator: torch.Generator, num_tokens: int
) -> torch.Tensor:
    """Random hidden states whose feature 0 is the constant beta offset input."""
    x = torch.randn(num_tokens, HIDDEN, generator=generator)
    x[:, 0] = 1.0
    return x


@pytest.fixture(scope="module")
def initial_pools() -> tuple[np.ndarray, np.ndarray]:
    """Non-zero starting pools, so the window and state carry-in are exercised.

    A real prefill starts from zeros, which is the weaker case: it cannot tell
    a dropped carry-in from a correct one.
    """
    generator = np.random.default_rng(7)
    conv = generator.standard_normal((MAX_SLOTS, CONV_DIM, KERNEL - 1)).astype(
        np.float32
    )
    recurrent = (
        generator.standard_normal((MAX_SLOTS, HEADS, HEAD_DIM, HEAD_DIM)) * 0.1
    ).astype(np.float32)
    return conv, recurrent


@pytest.fixture(scope="module")
def hidden_states() -> torch.Tensor:
    """``[sum(PREFILL_LENGTHS), HIDDEN]`` packed, as the decoder layer hands it."""
    generator = torch.Generator().manual_seed(11)
    return torch.randn(sum(PREFILL_LENGTHS), HIDDEN, generator=generator)


def _load(
    session: InferenceSession, weights: dict[str, torch.Tensor], name: str
) -> Model:
    layer = _KdaGraph()
    layer.load_state_dict(weights)
    graph = Graph(name, layer, input_types=_input_types())
    return session.load(graph, weights_registry=layer.state_dict())


@pytest.fixture(scope="module")
def decode_model(
    session: InferenceSession, weights: dict[str, torch.Tensor]
) -> Model:
    return _load(session, weights, "Glm5NextKdaDecode")


@pytest.fixture(scope="module")
def carry_model(
    session: InferenceSession, carry_weights: dict[str, torch.Tensor]
) -> Model:
    return _load(session, carry_weights, "Glm5NextKdaCarry")


def _run(
    model: Model,
    x: torch.Tensor,
    lengths: Sequence[int],
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Runs one call and returns ``(out, conv_pool_after, recurrent_pool_after)``."""
    device = model.input_devices[0]
    row_offsets = np.concatenate(
        [[0], np.cumsum(np.asarray(lengths, dtype=np.uint32))]
    ).astype(np.uint32)
    conv_buf = md.Buffer.from_numpy(initial_pools[0].copy()).to(device)
    rec_buf = md.Buffer.from_numpy(initial_pools[1].copy()).to(device)
    outputs = model.execute(
        md.Buffer.from_numpy(x.contiguous().numpy()).to(device),
        conv_buf,
        rec_buf,
        md.Buffer.from_numpy(np.asarray(SLOTS, dtype=np.int32)).to(device),
        md.Buffer.from_numpy(row_offsets).to(device),
    )
    return (
        torch.from_dlpack(outputs[0]).cpu(),
        torch.from_dlpack(conv_buf).cpu().numpy(),
        torch.from_dlpack(rec_buf).cpu().numpy(),
    )


def _reference(
    x: torch.Tensor,
    lengths: Sequence[int],
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
    norm_fn: GatedNorm | None = None,
    work_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Runs the reference per sequence and reassembles the packed output."""
    weights = as_dtype(reference_weights, work_dtype)
    x = x.to(work_dtype)
    conv_pool, recurrent_pool = (
        p.astype(_NUMPY_DTYPE[work_dtype]) for p in initial_pools
    )
    outs: list[torch.Tensor] = []
    start = 0
    for length, slot in zip(lengths, SLOTS, strict=True):
        out, conv_state, recurrent_state = kda_layer(
            x[start : start + length],
            weights,
            torch.from_numpy(conv_pool[slot]),
            torch.from_numpy(recurrent_pool[slot]),
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            rms_norm_eps=RMS_NORM_EPS,
            lower_bound=LOWER_BOUND,
            work_dtype=work_dtype,
            norm_fn=norm_fn,
        )
        outs.append(out)
        conv_pool[slot] = conv_state.numpy()
        recurrent_pool[slot] = recurrent_state.numpy()
        start += length
    return torch.cat(outs), conv_pool, recurrent_pool


@pytest.fixture(scope="module")
def prefill_max(
    decode_model: Model,
    hidden_states: torch.Tensor,
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    return _run(decode_model, hidden_states, PREFILL_LENGTHS, initial_pools)


@pytest.fixture(scope="module")
def prefill_reference(
    hidden_states: torch.Tensor,
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    return _reference(
        hidden_states, PREFILL_LENGTHS, reference_weights, initial_pools
    )


@pytest.fixture(scope="module")
def prefill_reference64(
    hidden_states: torch.Tensor,
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """The same formula in float64: the closest thing here to the exact answer."""
    return _reference(
        hidden_states,
        PREFILL_LENGTHS,
        reference_weights,
        initial_pools,
        work_dtype=torch.float64,
    )


def test_layer_matches_reference(
    prefill_max: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference64: tuple[torch.Tensor, np.ndarray, np.ndarray],
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> None:
    """Output and both pools, on a ragged two-sequence prefill.

    The float64 arm is what makes the tolerance an argument rather than a
    number. It establishes two things the budget rests on: that the reference
    transcription is arithmetically sound in float32, and how much of the gap
    to MAX is matmul precision. What actually catches a wrong layer is the
    three differential tests below, whose margins over this budget are
    measured, not assumed.
    """
    max_out, max_conv, max_recurrent = prefill_max
    ref_out = prefill_reference[0]
    exact_out, exact_conv, exact_recurrent = prefill_reference64

    assert max_out.shape == (sum(PREFILL_LENGTHS), HIDDEN)
    assert torch.isfinite(max_out).all()

    max_error = _rel_rms(exact_out, max_out)
    float32_error = _rel_rms(exact_out, ref_out)
    print(
        f"relative RMS against the float64 reference: MAX {max_error:.3e}, "
        f"float32 reference {float32_error:.3e}, budget {LAYER_REL_RMS:.3e}"
    )
    assert float32_error < REFERENCE_REL_RMS, (
        f"the float32 reference is itself {float32_error} from its float64 "
        "evaluation, so it cannot arbitrate anything at this scale"
    )
    assert max_error < LAYER_REL_RMS, (
        f"MAX is {max_error} from the float64 reference in relative RMS, past "
        f"the {LAYER_REL_RMS} matmul-precision budget"
    )

    for slot in SLOTS:
        for name, mine, exact in (
            ("conv", max_conv[slot], exact_conv[slot]),
            ("recurrent", max_recurrent[slot], exact_recurrent[slot]),
        ):
            error = _rel_rms(
                torch.from_numpy(exact), torch.from_numpy(mine).double()
            )
            assert error < LAYER_REL_RMS, (
                f"{name} pool slot {slot} is {error} from the float64 reference"
            )

    for slot in set(range(MAX_SLOTS)) - set(SLOTS):
        np.testing.assert_array_equal(max_conv[slot], initial_pools[0][slot])
        np.testing.assert_array_equal(
            max_recurrent[slot], initial_pools[1][slot]
        )


def test_decode_step_matches_reference(
    decode_model: Model,
    reference_weights: KdaWeights,
    prefill_max: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference: tuple[torch.Tensor, np.ndarray, np.ndarray],
) -> None:
    """One token per sequence, continuing from the prefill's pools.

    The interesting part is the carry-in: the conv window and the recurrent
    state written by the prefill are the only things distinguishing this from a
    fresh one-token sequence. Each side continues from its own prefill's pools,
    so this measures the step, not the prefill's accumulated error.
    """
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(len(SLOTS), HIDDEN, generator=generator)

    max_out, max_conv, max_recurrent = _run(
        decode_model, x, [1] * len(SLOTS), (prefill_max[1], prefill_max[2])
    )
    ref_out, ref_conv, ref_recurrent = _reference(
        x,
        [1] * len(SLOTS),
        reference_weights,
        (prefill_reference[1], prefill_reference[2]),
    )

    assert _rel_rms(ref_out, max_out) < LAYER_REL_RMS
    for slot in SLOTS:
        assert (
            _rel_rms(
                torch.from_numpy(ref_conv[slot]),
                torch.from_numpy(max_conv[slot]),
            )
            < LAYER_REL_RMS
        )
        assert (
            _rel_rms(
                torch.from_numpy(ref_recurrent[slot]),
                torch.from_numpy(max_recurrent[slot]),
            )
            < LAYER_REL_RMS
        )


def test_decode_continues_a_multi_chunk_prefill(
    carry_model: Model,
    carry_weights: dict[str, torch.Tensor],
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> None:
    """Float32 decode steps after a prefill that spans several chunks.

    :data:`PREFILL_LENGTHS` fit inside one 64-token chunk; these lengths span
    several, with ragged tails. Each side decodes from the pools its own
    prefill left, so whatever error the prefill wrote into the state is
    carried into every step. On gfx950 that prefill is the bf16 chunk path and
    the steps are float32 ``kda_decode``.

    The weights decay slowly and write weakly, so the state carries across
    chunks and steps. The reference decoding the same steps from a zeroed
    recurrent state shows it: it has to land far outside the tolerance.
    """
    reference_weights = _reference_weights(carry_weights)
    lengths = MULTI_CHUNK_PREFILL_LENGTHS
    generator = torch.Generator().manual_seed(17)
    x = _carry_hidden_states(generator, sum(lengths))
    max_out, max_conv, max_recurrent = _run(
        carry_model, x, lengths, initial_pools
    )
    ref_out, ref_conv, ref_recurrent = _reference(
        x, lengths, reference_weights, initial_pools
    )
    error = _rel_rms(ref_out, max_out)
    print(f"multi-chunk prefill output relative RMS {error:.3e}")
    assert error < LAYER_REL_RMS

    dropped_recurrent = ref_recurrent.copy()
    dropped_recurrent[list(SLOTS)] = 0.0
    dropped_conv = ref_conv
    decode_lengths = [1] * len(SLOTS)
    for step in range(CONTINUATION_DECODE_STEPS):
        x = _carry_hidden_states(generator, len(SLOTS))
        max_out, max_conv, max_recurrent = _run(
            carry_model, x, decode_lengths, (max_conv, max_recurrent)
        )
        ref_out, ref_conv, ref_recurrent = _reference(
            x, decode_lengths, reference_weights, (ref_conv, ref_recurrent)
        )
        dropped_out, dropped_conv, dropped_recurrent = _reference(
            x,
            decode_lengths,
            reference_weights,
            (dropped_conv, dropped_recurrent),
        )
        error = _rel_rms(ref_out, max_out)
        drift = _rel_rms(ref_out, dropped_out)
        print(
            f"decode step {step} output relative RMS {error:.3e}, "
            f"from a zeroed carried state {drift:.3e}"
        )
        assert error < LAYER_REL_RMS, f"decode step {step}: {error}"
        assert drift > WRONG_VARIANT_MIN_REL_DIFF, (
            f"decode step {step} barely depends on the carried state "
            f"({drift}), so this check would not catch it being dropped"
        )

    for slot in SLOTS:
        for name, mine, ref in (
            ("conv", max_conv, ref_conv),
            ("recurrent", max_recurrent, ref_recurrent),
        ):
            error = _rel_rms(
                torch.from_numpy(ref[slot]), torch.from_numpy(mine[slot])
            )
            print(f"{name} pool slot {slot} relative RMS {error:.3e}")
            assert error < LAYER_REL_RMS, (
                f"{name} pool slot {slot} is {error} from the reference"
            )
        drift = _rel_rms(
            torch.from_numpy(ref_recurrent[slot]),
            torch.from_numpy(dropped_recurrent[slot]),
        )
        print(
            f"recurrent pool slot {slot} from a zeroed carried state "
            f"relative RMS {drift:.3e}"
        )
        assert drift > WRONG_VARIANT_MIN_REL_DIFF, (
            f"zeroing the carried state barely moves recurrent pool slot "
            f"{slot} ({drift}), so this check would not catch it being dropped"
        )


@pytest.mark.parametrize("permutation", [(0, 2, 1), (2, 1, 0), (1, 2, 0)])
def test_conv_concat_order_is_load_bearing(
    prefill_max: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference: tuple[torch.Tensor, np.ndarray, np.ndarray],
    hidden_states: torch.Tensor,
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
    permutation: tuple[int, int, int],
) -> None:
    """Permuting the conv weight's q/k/v blocks changes the output.

    The placebo the verification ladder asks for. The layer concatenates its
    three projections in q, k, v order to match the weight adapter's
    ``Concatenate(dim=0)`` over ``[q_conv1d, k_conv1d, v_conv1d]``; at TP1 every
    permutation has the same shapes, so nothing but a numeric check can tell
    them apart.
    """
    blocks = reference_weights.conv1d.split(QKV_DIM, dim=0)
    permuted = dataclasses.replace(
        reference_weights,
        conv1d=torch.cat([blocks[i] for i in permutation], dim=0),
    )
    wrong_out, _, _ = _reference(
        hidden_states, PREFILL_LENGTHS, permuted, initial_pools
    )
    drift = _rel_rms(prefill_reference[0], wrong_out)
    assert drift > WRONG_VARIANT_MIN_REL_DIFF, (
        f"permutation {permutation} barely moves the output ({drift}), so this "
        "check would not catch a scrambled conv concat"
    )
    assert _rel_rms(prefill_reference[0], prefill_max[0]) < LAYER_REL_RMS


def test_gated_norm_normalizes_before_gating(
    prefill_max: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference: tuple[torch.Tensor, np.ndarray, np.ndarray],
    hidden_states: torch.Tensor,
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> None:
    """``o_norm`` normalizes, scales, then gates -- not gate-then-normalize.

    This is the ordering MAX's fused ``gated_group_rmsnorm`` gets the other way
    round, so reusing that kernel here would fail this test rather than pass
    silently.
    """
    wrong_out, _, _ = _reference(
        hidden_states,
        PREFILL_LENGTHS,
        reference_weights,
        initial_pools,
        norm_fn=gate_then_norm,
    )
    drift = _rel_rms(prefill_reference[0], wrong_out)
    assert drift > WRONG_VARIANT_MIN_REL_DIFF, (
        f"gate-then-normalize barely moves the output ({drift})"
    )
    assert _rel_rms(prefill_reference[0], prefill_max[0]) < LAYER_REL_RMS


def test_forget_gate_is_per_channel(
    prefill_max: tuple[torch.Tensor, np.ndarray, np.ndarray],
    prefill_reference: tuple[torch.Tensor, np.ndarray, np.ndarray],
    hidden_states: torch.Tensor,
    reference_weights: KdaWeights,
    initial_pools: tuple[np.ndarray, np.ndarray],
) -> None:
    """A per-head decay is a different layer, so the fixture must reject it.

    Modelled by flattening ``dt_bias`` within each head to its head mean, which
    is the closest a per-head gate can get to this one. If this drift were
    small, every other check here would be blind to the difference between KDA
    and Gated DeltaNet.
    """
    per_head = dataclasses.replace(
        reference_weights,
        dt_bias=reference_weights.dt_bias.view(HEADS, HEAD_DIM)
        .mean(-1, keepdim=True)
        .expand(HEADS, HEAD_DIM)
        .reshape(-1)
        .contiguous(),
    )
    wrong_out, _, _ = _reference(
        hidden_states, PREFILL_LENGTHS, per_head, initial_pools
    )
    drift = _rel_rms(prefill_reference[0], wrong_out)
    assert drift > WRONG_VARIANT_MIN_REL_DIFF, (
        f"a per-head decay barely moves the output ({drift})"
    )
    assert _rel_rms(prefill_reference[0], prefill_max[0]) < LAYER_REL_RMS


def test_replay_capture_records_the_pre_activation_gate(
    weights: dict[str, torch.Tensor],
) -> None:
    """What ``replay_capture`` hands a state rollback, at graph-build time.

    A rollback re-runs the two state kernels over the accepted rows instead of
    taking a second pass over the weights, so it needs the tensors that fed
    them. The load-bearing detail is *where* they are captured: the kernel
    folds in ``exp(A_log)``, the bounded sigmoid and beta's own sigmoid, so
    what is captured is the raw pre-activation gate -- per channel,
    ``[T, heads, head_dim]`` -- and not the computed per-head decay that
    Qwen3.5's equivalent captures (``[T, heads]``). Feeding a post-activation
    tensor back into ``kda_decode`` would apply both activations twice: it
    runs, and it answers slightly badly.

    No execution needed -- the shapes settle it.
    """
    layer = _KdaGraph()
    # `load_state_dict` is what qualifies the weight names; without it every
    # `Linear` still calls its weight "weight" and the graph rejects the second.
    layer.load_state_dict(weights)
    captured: list[KdaReplayInputs] = []
    layer.captured = captured
    Graph("kda_replay_capture", layer, input_types=_input_types())

    assert len(captured) == 1
    replay = captured[0]
    assert replay.qkv.shape[1] == CONV_DIM
    assert replay.conv_weight.shape == [CONV_DIM, KERNEL]
    # Per channel: one axis wider than a per-head decay would be.
    assert replay.raw_gate.rank == 3
    assert replay.raw_gate.shape[1] == HEADS
    assert replay.raw_gate.shape[2] == HEAD_DIM
    assert replay.beta_logits.rank == 2
    assert replay.beta_logits.shape[1] == HEADS
    # Every row shares the packed token axis the layer was called with.
    tokens = replay.qkv.shape[0]
    assert replay.raw_gate.shape[0] == tokens
    assert replay.beta_logits.shape[0] == tokens
    # The recurrence op consumes these in float32 regardless of `dtype`.
    for tensor in replay:
        assert tensor.dtype == DType.float32
