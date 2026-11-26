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
"""CPU tests for per-row verdict guards and masking."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops.sampling import (
    AcceptanceSampler,
    _cpu_greedy_verdict,
    _reshape_target_logits,
)

B = 2
K = 3
VOCAB = 8
PEAK_TOKEN = 3
PEAK_LOGIT = 20.0


def _length_guard_graph() -> Graph:
    device = DeviceRef.GPU(0)
    return Graph(
        "verdict_lengths",
        input_types=[
            TensorType(DType.int64, ["batch_size", "num_steps"], device),
            TensorType(
                DType.float32, ["total_output_len", "vocab_size"], device
            ),
            TensorType(DType.int32, ["batch_size"], device),
        ],
    )


def test_lengths_on_an_argmax_sampler_are_refused() -> None:
    """The argmax family masks draft ids instead, so lengths raise."""
    with _length_guard_graph() as graph:
        drafts, logits, lengths = (v.tensor for v in graph.inputs)
        sampler = AcceptanceSampler(
            num_draft_steps=K,
            use_stochastic=False,
            draft_proposal="argmax",
        )
        with pytest.raises(ValueError, match="verify_lengths"):
            sampler(drafts, logits, verify_lengths=lengths)
        graph.output()


def test_lengths_on_a_synthetic_sampler_are_refused() -> None:
    """The synthetic verdict draws from a uniform step count, so lengths
    raise rather than being dropped."""
    with _length_guard_graph() as graph:
        drafts, logits, lengths = (v.tensor for v in graph.inputs)
        sampler = AcceptanceSampler(
            num_draft_steps=K,
            use_stochastic=True,
            synthetic_acceptance_rate=0.5,
            draft_proposal="sampled",
            vocab_size=VOCAB,
        )
        with pytest.raises(ValueError, match="synthetic_acceptance_rate"):
            sampler(
                drafts,
                logits,
                seed=ops.constant(0, DType.uint64, drafts.device),
                verify_lengths=lengths,
            )
        graph.output()


def test_the_cpu_greedy_verdict_keeps_each_rows_length() -> None:
    """The CPU greedy override masks positions past k_r, like the strict
    path. Runs on the CPU device; the sampled verdict itself is
    GPU-only, so no public CPU path reaches the override."""
    device = DeviceRef.CPU()
    with Graph(
        "cpu_greedy_lengths",
        input_types=[
            TensorType(
                DType.float32, ["total_output_len", "vocab_size"], device
            ),
            TensorType(DType.int64, ["batch_size", "num_steps"], device),
            TensorType(DType.int32, ["batch_size"], device),
        ],
    ) as graph:
        logits, drafts, lengths = (v.tensor for v in graph.inputs)
        rejected, _recovered, _bonus = _cpu_greedy_verdict(
            _reshape_target_logits(logits),
            drafts,
            "batch_size",
            DType.int64,
            lengths,
        )
        graph.output(rejected)
    model = InferenceSession(devices=[CPU()]).load(graph)
    # Every draft matches the target argmax, so without the mask each
    # row would accept all K positions.
    logits_np = np.zeros((B * (K + 1), VOCAB), dtype=np.float32)
    logits_np[:, PEAK_TOKEN] = PEAK_LOGIT
    (out,) = model.execute(
        Buffer.from_numpy(logits_np),
        Buffer.from_numpy(np.full((B, K), PEAK_TOKEN, dtype=np.int64)),
        Buffer.from_numpy(np.array([1, K], dtype=np.int32)),
    )
    assert isinstance(out, Buffer)
    rejected_np = out.to_numpy()
    # Row 0 rejects at and past k_r = 1; row 1 never rejects.
    np.testing.assert_array_equal(rejected_np[0], [False, True, True])
    np.testing.assert_array_equal(rejected_np[1], [False, False, False])
