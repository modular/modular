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
"""Per-row verify lengths inside the sampled-draft verdict.

The sampled verdict gathers ``p`` and ``q`` at the draft id, so a masked id
cannot exclude a position the way it does for the argmax family. A row with
``k_r < K`` must stop at ``k_r``: the position rejects and its token is drawn
from the target's own distribution, with no draft mass subtracted.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops.sampling import AcceptanceSampler
from max.pipelines.speculative.unified_graph_ops import (
    accept_and_pick_next_tokens,
)
from spec_step_stubs import outputs_to_numpy, unrestricted_sampling_buffers

B = 2
K = 3
VOCAB = 8
PEAK_TOKEN = 3
PEAK_LOGIT = 20.0


def _target_logits() -> np.ndarray:
    """Every position's logits peak hard at ``PEAK_TOKEN``."""
    logits = np.zeros((B * (K + 1), VOCAB), dtype=np.float32)
    logits[:, PEAK_TOKEN] = PEAK_LOGIT
    return logits


def _draft_dists() -> np.ndarray:
    """One-hot at the drafted token, so ``q = p`` up to the peak's mass."""
    dists = np.zeros((B, K, VOCAB), dtype=np.float32)
    dists[:, :, PEAK_TOKEN] = 1.0
    return dists


def _build(verify_lengths: bool, fixed_width: bool = False) -> Model:
    """One accept step over the sampled knobs, with or without lengths.

    ``fixed_width`` declares the drafts ``[batch, K]``, as the per-row driver
    rebinds them, while the logits and distributions stay symbolic.
    """
    device = DeviceRef.GPU(0)
    draft_steps: int | str = K if fixed_width else "num_steps"
    input_types = [
        TensorType(DType.int64, ["batch_size", draft_steps], device),
        TensorType(DType.float32, ["total_output_len", "vocab_size"], device),
        TensorType(DType.uint64, ["batch_size"], device),
        TensorType(DType.float32, ["batch_size"], device),
        TensorType(DType.int64, ["batch_size"], device),
        TensorType(DType.int64, [], DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device),
        TensorType(DType.float32, [], DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size", "num_steps", VOCAB], device),
        TensorType(DType.int32, ["batch_size"], device),
    ]
    with Graph("sampled_per_row", input_types=input_types) as graph:
        (
            drafts_in,
            logits_in,
            seed_in,
            temperature_in,
            top_k_in,
            max_k_in,
            top_p_in,
            min_top_p_in,
            dists_in,
            lengths_in,
        ) = (v.tensor for v in graph.inputs)
        sampler = AcceptanceSampler(
            num_draft_steps=K,
            use_stochastic=True,
            draft_proposal="sampled",
            vocab_size=VOCAB,
        )
        num_accepted, _recovered, _bonus, next_tokens = (
            accept_and_pick_next_tokens(
                sampler,
                drafts_in,
                logits_in,
                seed=seed_in,
                temperature=temperature_in,
                top_k=top_k_in,
                max_k=max_k_in,
                top_p=top_p_in,
                min_top_p=min_top_p_in,
                draft_probs_full=dists_in,
                verify_lengths=lengths_in if verify_lengths else None,
            )
        )
        graph.output(num_accepted, next_tokens)
    session = InferenceSession(devices=[Accelerator()])
    return session.load(graph)


def _execute(
    model: Model, lengths: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    gpu = Accelerator()
    drafts = np.full((B, K), PEAK_TOKEN, dtype=np.int64)
    buffers = [
        Buffer.from_numpy(drafts).to(gpu),
        Buffer.from_numpy(_target_logits()).to(gpu),
        *unrestricted_sampling_buffers(B, VOCAB, gpu),
        Buffer.from_numpy(_draft_dists()).to(gpu),
    ]
    if lengths is not None:
        buffers.append(
            Buffer.from_numpy(np.asarray(lengths, dtype=np.int32)).to(gpu)
        )
    num_accepted, next_tokens = outputs_to_numpy(model.execute(*buffers))
    return num_accepted.reshape(-1), next_tokens.reshape(-1)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="sampled verdict is GPU-only"
)
def test_a_sampled_per_row_verify_stops_at_each_rows_length() -> None:
    """Lengths ``[1, 3]`` accept exactly one and all three drafts."""
    num_accepted, _ = _execute(_build(True), np.array([1, 3]))
    np.testing.assert_array_equal(num_accepted, [1, 3])


@pytest.mark.skipif(
    accelerator_count() == 0, reason="sampled verdict is GPU-only"
)
def test_the_draw_past_a_rows_length_comes_from_the_target() -> None:
    """Row 0's next token is the peak, drawn from ``p`` with no ``q``
    subtracted; with ``q`` subtracted the residual has no mass there."""
    _accepted, next_tokens = _execute(_build(True), np.array([1, 3]))
    assert next_tokens[0] == PEAK_TOKEN


@pytest.mark.skipif(
    accelerator_count() == 0, reason="sampled verdict is GPU-only"
)
def test_a_fixed_width_per_row_verify_builds_and_keeps_lengths() -> None:
    """Drafts declared at a fixed ``K`` build against symbolic logits and
    still stop each row at its own length."""
    num_accepted, next_tokens = _execute(
        _build(True, fixed_width=True), np.array([1, 3])
    )
    np.testing.assert_array_equal(num_accepted, [1, 3])
    assert next_tokens[0] == PEAK_TOKEN


@pytest.mark.skipif(
    accelerator_count() == 0, reason="sampled verdict is GPU-only"
)
def test_without_lengths_the_sampled_verdict_is_unchanged() -> None:
    """The same inputs with no lengths accept every draft."""
    num_accepted, _ = _execute(_build(False), np.array([1, 3]))
    np.testing.assert_array_equal(num_accepted, [K, K])


@pytest.mark.skipif(
    accelerator_count() == 0, reason="sampled verdict is GPU-only"
)
def test_argmax_lengths_mask_the_drafts_past_each_row() -> None:
    """An argmax sampler through the shared accept masks ids past k_r."""
    device = DeviceRef.GPU(0)
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device),
        TensorType(DType.float32, ["total_output_len", "vocab_size"], device),
        TensorType(DType.uint64, ["batch_size"], device),
        TensorType(DType.float32, ["batch_size"], device),
        TensorType(DType.int64, ["batch_size"], device),
        TensorType(DType.int64, [], DeviceRef.CPU()),
        TensorType(DType.float32, ["batch_size"], device),
        TensorType(DType.float32, [], DeviceRef.CPU()),
        TensorType(DType.int32, ["batch_size"], device),
    ]
    with Graph("argmax_lengths", input_types=input_types) as graph:
        (
            drafts_in,
            logits_in,
            seed_in,
            temperature_in,
            top_k_in,
            max_k_in,
            top_p_in,
            min_top_p_in,
            lengths_in,
        ) = (v.tensor for v in graph.inputs)
        sampler = AcceptanceSampler(
            num_draft_steps=K,
            use_stochastic=True,
            draft_proposal="argmax",
        )
        num_accepted, _recovered, _bonus, _next = accept_and_pick_next_tokens(
            sampler,
            drafts_in,
            logits_in,
            seed=seed_in,
            temperature=temperature_in,
            top_k=top_k_in,
            max_k=max_k_in,
            top_p=top_p_in,
            min_top_p=min_top_p_in,
            verify_lengths=lengths_in,
        )
        graph.output(num_accepted)
    session = InferenceSession(devices=[Accelerator()])
    model = session.load(graph)
    buffers = [
        Buffer.from_numpy(np.full((B, K), PEAK_TOKEN, dtype=np.int64)).to(
            Accelerator()
        ),
        Buffer.from_numpy(_target_logits()).to(Accelerator()),
        Buffer.from_numpy(np.arange(B, dtype=np.uint64)).to(Accelerator()),
        Buffer.from_numpy(np.ones(B, dtype=np.float32)).to(Accelerator()),
        Buffer.from_numpy(np.full(B, VOCAB, dtype=np.int64)).to(Accelerator()),
        Buffer.from_numpy(np.array(VOCAB, dtype=np.int64)),
        Buffer.from_numpy(np.ones(B, dtype=np.float32)).to(Accelerator()),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
        Buffer.from_numpy(np.array([1, 3], dtype=np.int32)).to(Accelerator()),
    ]
    (out,) = model.execute(*buffers)
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(out.to_numpy().reshape(-1), [1, 3])
