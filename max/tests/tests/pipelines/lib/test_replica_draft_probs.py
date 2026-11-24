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
"""The overlap pipeline keeps per-replica draft distributions in their slots.

Under ``draft_probs_layout="replica"`` each data-parallel replica's proposal
distributions live in its own persistent buffer. Every step clears the rows
it binds, then carries each request's previous distribution into its new
slot, so a slot reused by a preempted, resumed or new request never reads a
stale distribution.
"""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import Buffer
from max.pipelines.lib import OverlapTextGenerationPipeline
from max.pipelines.lib.interfaces.pipeline_model import UnifiedEagleOutputs
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _OOB_IDX,
    _carry_replica_draft_probs,
    _sampled_draft_vocab_size,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
)
from max.pipelines.speculative.config import SpeculativeConfig

K = 3
VOCAB = 4
MAX_BATCH = 4


def _buffer(values: np.ndarray) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _pipeline(stale: list[np.ndarray]) -> Any:
    state = SimpleNamespace(
        replica_draft_probs=[_buffer(s) for s in stale],
        replica_draft_probs_zero_rows=[
            _buffer(np.zeros((K, VOCAB))) for _ in stale
        ],
        persistent_draft_probs_full=None,
    )
    return SimpleNamespace(_spec_decode_state=state)


@pytest.mark.parametrize("width", [K, K - 1])
def test_carried_rows_follow_their_requests_and_new_rows_are_zero(
    width: int,
) -> None:
    rng = np.random.default_rng(0)
    stale = [rng.random((MAX_BATCH, K, VOCAB)) + 1 for _ in range(2)]
    pipeline = _pipeline(stale)
    views = OverlapTextGenerationPipeline._prepare_draft_probs_full(
        pipeline, [3, 2], width
    )
    assert isinstance(views, list)
    prev = [rng.random((rows, K, VOCAB), dtype=np.float32) for rows in (3, 2)]
    # Replica 0: prev row 1 left (preempted), rows 2 and 0 swapped slots, and
    # slot 2 holds a request the previous step did not run. Replica 1 keeps
    # its order, so its rows move together.
    prev_to_curr = np.array([1, _OOB_IDX, 0, 3, 4], dtype=np.int64)
    _carry_replica_draft_probs(
        [_buffer(p) for p in prev], views, prev_to_curr, [3, 2], [3, 2]
    )
    r0, r1 = (view.to_numpy() for view in views)
    np.testing.assert_array_equal(r0[0], prev[0][2, :width])
    np.testing.assert_array_equal(r0[1], prev[0][0, :width])
    np.testing.assert_array_equal(r0[2], 0)
    np.testing.assert_array_equal(r1, prev[1][:, :width])


def test_a_row_cannot_change_replicas() -> None:
    views = OverlapTextGenerationPipeline._prepare_draft_probs_full(
        _pipeline([np.zeros((MAX_BATCH, K, VOCAB))] * 2), [1, 1], K
    )
    assert isinstance(views, list)
    prev = [_buffer(np.ones((1, K, VOCAB))), _buffer(np.ones((1, K, VOCAB)))]
    with pytest.raises(AssertionError, match="moved between replicas"):
        _carry_replica_draft_probs(
            prev, views, np.array([1, _OOB_IDX], np.int64), [1, 1], [1, 1]
        )


def test_outputs_name_one_distribution_or_one_per_replica() -> None:
    tokens = [_buffer(np.zeros(1)) for _ in range(3)]
    dists = [_buffer(np.zeros((1, K, VOCAB))) for _ in range(2)]
    argmax = UnifiedEagleOutputs.from_buffers(tokens)
    assert argmax.next_draft_probs_full is None
    assert argmax.next_draft_probs_replicas is None
    device0 = UnifiedEagleOutputs.from_buffers([*tokens, dists[0]])
    assert device0.next_draft_probs_full is dists[0]
    assert device0.next_draft_probs_replicas is None
    replica = UnifiedEagleOutputs.from_buffers([*tokens, *dists])
    assert replica.next_draft_probs_full is None
    assert replica.next_draft_probs_replicas == dists


@pytest.mark.parametrize(
    ("layout", "distributions", "accepted"),
    [
        ("device0", 1, True),
        ("replica", 2, True),
        ("replica", 1, False),
        ("device0", 2, False),
    ],
)
def test_the_layout_must_match_the_graphs_distributions(
    layout: str, distributions: int, accepted: bool
) -> None:
    model = object.__new__(SpecDecodePipelineModel)
    model.model = cast(
        Any,
        SimpleNamespace(
            output_metadata=[SimpleNamespace(shape=[1])] * 3
            + [SimpleNamespace(shape=[1, K, VOCAB])] * distributions
        ),
    )
    speculative = SpeculativeConfig.model_validate(
        {
            "speculative_method": "dspark",
            "num_speculative_tokens": K,
            "draft_proposal": "sampled",
            "draft_head_parallelism": "replica",
            "draft_probs_layout": layout,
        }
    )
    if accepted:
        assert _sampled_draft_vocab_size(model, speculative) == VOCAB
    else:
        with pytest.raises(ValueError, match="draft_probs_layout"):
            _sampled_draft_vocab_size(model, speculative)
