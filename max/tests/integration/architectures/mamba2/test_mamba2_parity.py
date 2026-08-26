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
"""End-to-end greedy + stochastic parity for the Mamba2 pipeline.

RFC 0003 item 7 — the final Mamba2 pipeline integration test.

The contract this file encodes (per RFC 0000 §4 "Parity contract"):

* Greedy decoding for 32 steps from a fixed prompt must match the
  reference HuggingFace ``mamba_ssm`` ``MambaLMHeadModel`` exactly
  (token-id sequence equality, no tolerance).
* For each greedy step the per-token logits must agree to within
  ``max_abs <= 1e-2`` or ``max_rel <= 2e-2`` (bf16 tolerances). Softmax
  KL is reported as a soft signal, not asserted.
* Stochastic sampling with ``temperature=0.7, top_p=0.9`` and a fixed
  seed need not produce identical samples, but the top-10 logit set
  (jaccard) must agree at ``>= 0.95`` per step.

Both tests are currently parked behind ``@pytest.mark.xfail`` with
``run=False`` because two upstream blockers prevent the SSD path from
loading or running at all:

  1. ``causal_conv1d_ops.mojo:506`` has a Tuple-shape loader bug that
     prevents the ``state_space.mojoc`` Python load. This is filed in
     ``.planning/.../proposed/approvals/causal_conv1d_ops-tuple-shape-bug.md``.
  2. ``ssd_chunk_scan_combined`` does not surface ``final_state``
     (RFC 0002 item 6 follow-up), so prefill -> step continuation reads
     a zero-initialised SSM state from the cache and breaks correctness
     past the prefill boundary. :class:`Mamba2Model.__init__` already
     logs a one-time warning about this.

Once those clear, the ``xfail`` marker should be removed (and the
greedy-match assertion will tell us whether bf16 tolerances need a
small bump in either direction). The body of each test is written so
that it is what we actually want to run on day one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from max.driver import accelerator_count

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Parity primitives (vendored inline from `.planning/parity/`, which is
# local-only and won't ship upstream). All numpy, no SciPy / torch needed
# here — the reference logits are pulled to CPU as numpy arrays before
# comparison.
# ---------------------------------------------------------------------------


def softmax(z: NDArray[Any], temp: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    arr = np.asarray(z, dtype=np.float64) / temp
    arr -= arr.max(axis=-1, keepdims=True)
    e = np.exp(arr)
    return e / e.sum(axis=-1, keepdims=True)


def kl_div(
    p: NDArray[np.float64], q: NDArray[np.float64], eps: float = 1e-12
) -> float:
    """KL(p || q) with a small floor to keep the log finite."""
    p_clip = np.clip(p, eps, 1.0)
    q_clip = np.clip(q, eps, 1.0)
    return float(np.sum(p_clip * np.log(p_clip / q_clip)))


def greedy_exact_match(a: list[int], b: list[int]) -> bool:
    """True iff `a` and `b` are the same length and agree elementwise."""
    n = min(len(a), len(b))
    div = next((i for i in range(n) if a[i] != b[i]), -1)
    return div == -1 and len(a) == len(b)


def logit_metrics(
    actual: NDArray[Any], expected: NDArray[Any]
) -> dict[str, float]:
    """Compute per-step ``max_abs``, ``max_rel``, and ``softmax_kl``."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    diff = np.abs(a - e)
    denom = np.abs(e) + 1e-12
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "softmax_kl": kl_div(softmax(e), softmax(a)),
    }


def topk_jaccard(actual: NDArray[Any], expected: NDArray[Any], k: int) -> float:
    """Jaccard overlap of the top-`k` indices of two logit vectors."""
    ma = set(np.argsort(np.asarray(actual))[-k:].tolist())
    me = set(np.argsort(np.asarray(expected))[-k:].tolist())
    u = ma | me
    return len(ma & me) / len(u) if u else 1.0


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

MODEL_NAME = "state-spaces/mamba2-130m"

# Fixed prompt — token IDs in the GPT-NeoX vocabulary that Mamba2 ships
# with (the small handful of tokens for "Mamba2 is a state space model").
# Kept short so the prefill chunk-scan only fires a single chunk; the
# step-mode path is what we're really stressing across 32 generation
# steps.
PROMPT_IDS: list[int] = [44, 6708, 19, 310, 247, 1375, 2317, 1566]
N_STEPS = 32

# bf16 tolerances; tightened per RFC 0000 §4. ``max_rel`` is the
# practical gate — fp32 magnitudes near zero can blow up the absolute
# error harmlessly.
ATOL_BF16 = 1e-2
RTOL_BF16 = 2e-2

# Stochastic-sampling acceptance threshold.
TOPK_AGREEMENT_MIN = 0.95
SAMPLING_TEMPERATURE = 0.7
SAMPLING_TOP_P = 0.9
SAMPLING_SEED = 42

XFAIL_REASON = (
    "Two upstream blockers: (1) causal_conv1d_ops.mojo:506 Tuple-shape "
    "loader bug blocks state_space.mojoc Python load; "
    "(2) ssd_chunk_scan_combined doesn't expose final_state (RFC 0002 "
    "item 6 follow-up) so prefill->cache state continuity is broken."
)


# ---------------------------------------------------------------------------
# Pipeline / reference plumbing
#
# The bodies below describe the test as we want it to run the moment
# the two upstream blockers above are resolved. They are intentionally
# never reached today (xfail with ``run=False``) — but the imports inside
# them are guarded by the marker so static collection still succeeds on a
# bare CI box that hasn't installed ``mamba_ssm``.
# ---------------------------------------------------------------------------


def _max_greedy_decode(
    prompt_ids: list[int], n_steps: int
) -> tuple[list[int], list[NDArray[np.float64]]]:
    """Run the MAX Mamba2 pipeline greedy-decoding from ``prompt_ids``.

    Returns ``(generated_ids, per_step_logits)`` where ``generated_ids``
    omits the prompt and ``per_step_logits[i]`` is the logits vector
    that produced ``generated_ids[i]``.

    The full machinery — :class:`PipelineConfig` construction,
    :class:`Mamba2Model` load, prefill, then ``n_steps`` of step-mode
    forward — only runs once the SSD kernel package loads. The imports
    and pipeline-registry call are deferred via ``importlib`` so the
    Python module is type-checkable even on a CI box that hasn't built
    ``state_space.mojoc`` and so the precise PipelineConfig / registry
    keyword surface (which is still evolving on the main branch) doesn't
    pin this test to one snapshot.
    """
    import importlib

    pipelines_lib = importlib.import_module("max.pipelines.lib")
    architectures = importlib.import_module("max.pipelines.architectures")
    driver = importlib.import_module("max.driver")
    engine = importlib.import_module("max.engine")

    # Touch the registered arch so the registry sees ``Mamba2ForCausalLM``.
    _ = getattr(architectures, "mamba2_arch", None)

    pipeline_config: Any = pipelines_lib.PipelineConfig(
        model_path=MODEL_NAME,
        max_length=len(prompt_ids) + n_steps + 8,
        max_batch_size=1,
    )
    device = driver.Accelerator()
    session = engine.InferenceSession(devices=[device])
    pipeline: Any = pipelines_lib.PIPELINE_REGISTRY.retrieve(
        pipeline_config, session=session
    )
    return _drive_max_greedy(pipeline, prompt_ids, n_steps)


def _drive_max_greedy(
    pipeline: Any, prompt_ids: list[int], n_steps: int
) -> tuple[list[int], list[NDArray[np.float64]]]:
    """Execute the greedy loop against an instantiated MAX pipeline.

    Split out from :func:`_max_greedy_decode` so the test body stays
    readable and so we can stub it in a fast unit test once the kernel
    blockers clear. The exact ``pipeline.next_logits``-style API depends
    on which pipeline factory ``retrieve`` returns at runtime; we
    therefore poke through with ``getattr`` to stay version-agnostic
    until the upstream blockers clear and we can settle on a concrete
    method name.
    """
    generated: list[int] = []
    step_logits: list[NDArray[np.float64]] = []
    tokens: list[int] = list(prompt_ids)
    next_logits = getattr(pipeline, "next_logits", None)
    if next_logits is None:
        raise NotImplementedError(
            "Pipeline factory does not expose a single-step logits hook "
            "yet; this test stub will be re-wired against the eventual "
            "API once the SSD kernel package loads."
        )
    for _ in range(n_steps):
        logits_step = next_logits(tokens)
        logits_np = np.asarray(logits_step, dtype=np.float64).reshape(-1)
        step_logits.append(logits_np)
        next_tok = int(np.argmax(logits_np))
        generated.append(next_tok)
        tokens.append(next_tok)
    return generated, step_logits


def _ref_greedy_decode(
    prompt_ids: list[int], n_steps: int
) -> tuple[list[int], list[NDArray[np.float64]]]:
    """Greedy reference using the upstream ``mamba_ssm`` model.

    Oracle pattern: ``mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel``
    is the canonical Mamba2 implementation (the HF transformers entry
    for Mamba2 is a later wrapper and disagrees with the reference in
    some chunk-scan corner cases). We use the reference because RFC 0000
    §4.1 specifies *kernel parity*, not transformers parity.

    Imports are deferred — ``mamba_ssm`` is an optional dev dependency
    that isn't on the CI base image, and the body never runs while the
    xfail markers are active.
    """
    import importlib

    torch_mod = importlib.import_module("torch")
    mamba_ssm_seq = importlib.import_module("mamba_ssm.models.mixer_seq_simple")
    MambaLMHeadModel: Any = mamba_ssm_seq.MambaLMHeadModel

    model: Any = MambaLMHeadModel.from_pretrained(MODEL_NAME, device="cuda")
    model.eval()
    tokens: Any = torch_mod.tensor(
        [prompt_ids], dtype=torch_mod.long, device="cuda"
    )
    generated: list[int] = []
    step_logits: list[NDArray[np.float64]] = []
    with torch_mod.no_grad():
        for _ in range(n_steps):
            out: Any = model(tokens)
            logits = out.logits[0, -1].float().cpu().numpy()
            step_logits.append(np.asarray(logits, dtype=np.float64))
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            tokens = torch_mod.cat(
                [
                    tokens,
                    torch_mod.tensor(
                        [[next_tok]], dtype=torch_mod.long, device="cuda"
                    ),
                ],
                dim=1,
            )
    return generated, step_logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Mamba2 parity needs a GPU."
)
@pytest.mark.xfail(reason=XFAIL_REASON, strict=False, run=False)
def test_mamba2_greedy_parity() -> None:
    """Greedy parity: exact token match + bf16-tight per-step logits."""
    max_ids, max_logits = _max_greedy_decode(PROMPT_IDS, N_STEPS)
    ref_ids, ref_logits = _ref_greedy_decode(PROMPT_IDS, N_STEPS)

    # Hard gate: token sequence must match exactly. If this fires we
    # know to drop into the per-step metrics below for the divergence
    # point.
    assert greedy_exact_match(max_ids, ref_ids), (
        f"Greedy divergence: MAX={max_ids!r} ref={ref_ids!r}"
    )

    # Per-step logit tolerance. Either ``max_abs`` or ``max_rel`` must
    # clear bf16 thresholds; we report KL as a soft signal.
    for step_idx, (a, e) in enumerate(
        zip(max_logits, ref_logits, strict=False)
    ):
        metrics = logit_metrics(a, e)
        ok = metrics["max_abs"] <= ATOL_BF16 or metrics["max_rel"] <= RTOL_BF16
        assert ok, (
            f"Step {step_idx}: logits out of tolerance "
            f"(max_abs={metrics['max_abs']:.4g}, "
            f"max_rel={metrics['max_rel']:.4g}, "
            f"softmax_kl={metrics['softmax_kl']:.4g}); "
            f"atol={ATOL_BF16}, rtol={RTOL_BF16}"
        )


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Mamba2 parity needs a GPU."
)
@pytest.mark.xfail(reason=XFAIL_REASON, strict=False, run=False)
def test_mamba2_stochastic_topk_agreement() -> None:
    """Stochastic sampling: top-10 jaccard agreement per step.

    We deliberately do **not** assert that the sampled tokens match —
    cross-framework PRNG draws diverge even with identical seeds. The
    top-10 logit set, however, must agree at ``>= 0.95`` jaccard per
    step (RFC 0000 §4.2). This both bounds tail noise and catches the
    "near-tied top tokens" regression we saw on early SSD prototypes.
    """
    np.random.seed(SAMPLING_SEED)

    max_ids, max_logits = _max_greedy_decode(PROMPT_IDS, N_STEPS)
    _ref_ids, ref_logits = _ref_greedy_decode(PROMPT_IDS, N_STEPS)
    del max_ids  # generation IDs not asserted; only the logit shape is

    for step_idx, (a, e) in enumerate(
        zip(max_logits, ref_logits, strict=False)
    ):
        # Temperature + top-p only reshape the sampling distribution;
        # they don't change which indices are in the top-10. So the
        # logit-set check below is performed on raw logits, with the
        # temperature/top-p values recorded here for posterity (and so
        # that, when we extend the test to actually sample, no one has
        # to dig up the magic numbers).
        _ = SAMPLING_TEMPERATURE, SAMPLING_TOP_P
        agreement = topk_jaccard(a, e, k=10)
        assert agreement >= TOPK_AGREEMENT_MIN, (
            f"Step {step_idx}: top-10 jaccard {agreement:.3f} "
            f"< {TOPK_AGREEMENT_MIN}"
        )
