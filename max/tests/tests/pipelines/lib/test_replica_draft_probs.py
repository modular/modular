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

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max._artifact.value_types import treedef_to_json
from max.driver import Buffer
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import ModuleList
from max.pipelines.lib import OverlapTextGenerationPipeline
from max.pipelines.lib.interfaces.pipeline_model import UnifiedEagleOutputs
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _OOB_IDX,
    _carry_replica_draft_probs,
    _sampled_draft_vocab_size,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _DriverCall,
    _SpecDecodeModule,
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    spec_decode_outputs,
)

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


def test_score_on_buffers_do_not_populate_the_distributions() -> None:
    """A score-on export's fourth output is the scores, ahead of any
    distribution: skipping it keeps the distribution fields empty, where
    the positional read would have stored the scores as one
    distribution."""
    tokens = [_buffer(np.zeros(1)) for _ in range(3)]
    scores = _buffer(np.zeros((1, K)))
    dists = [_buffer(np.zeros((1, K, VOCAB))) for _ in range(2)]
    score_on = UnifiedEagleOutputs.from_buffers(
        [*tokens, scores, *dists], skip=1
    )
    assert score_on.next_draft_probs_full is None
    assert score_on.next_draft_probs_replicas == dists


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
    model.pipeline_config = cast(Any, SimpleNamespace(speculative=speculative))
    if accepted:
        assert _sampled_draft_vocab_size(model, speculative) == VOCAB
    else:
        with pytest.raises(ValueError, match="draft_probs_layout"):
            _sampled_draft_vocab_size(model, speculative)


@pytest.mark.parametrize(
    "owner", [SpecDecodePipelineModel, _UnifiedSpecDecodeModelMixin]
)
def test_confidence_outputs_are_not_draft_distributions(owner: type) -> None:
    """A per-row verify's extra outputs are the scores and the lengths,
    both ahead of any distribution; the distribution probe skips them,
    or it misreads the scores' trailing K as a vocabulary and refuses an
    argmax export at pipeline construction. Both property owners are
    proven — the mixin's copy serves every unified spec-decode package."""
    model: Any = object.__new__(owner)
    outputs = (
        [SimpleNamespace(shape=[1])] * 3
        + [SimpleNamespace(shape=[1, K])]
        + [SimpleNamespace(shape=[1])]
    )
    model.model = cast(Any, SimpleNamespace(output_metadata=outputs))
    model.pipeline_config = cast(
        Any,
        SimpleNamespace(
            speculative=SpeculativeConfig.model_validate(
                {
                    "speculative_method": "dspark",
                    "num_speculative_tokens": K,
                    "per_row_verify": True,
                }
            )
        ),
    )
    assert model.sampled_draft_vocab_size is None
    assert model.num_draft_distributions == 0


def test_spec_decode_outputs_name_scores_and_lengths_under_their_flags() -> (
    None
):
    """The spec output group's fourth member is the drafts' head-free
    scores under a per-row verify, the fifth is each row's verify length;
    without the flags the same fourth member is the draft distributions."""
    with Graph("spec_output_names", input_types=[]) as graph:
        cpu = DeviceRef.CPU()

        def rows(dtype: DType, shape: tuple[int, ...]) -> TensorValue:
            return ops.constant(
                np.zeros(shape, dtype=dtype.to_numpy()), dtype, cpu
            )

        base = [
            rows(DType.int32, (2,)),
            rows(DType.int64, (2,)),
            rows(DType.int64, (2, K)),
        ]
        scores = rows(DType.float32, (2, K))
        probs = rows(DType.float32, (2, K, VOCAB))

        plain = spec_decode_outputs(base)
        assert list(plain["spec"]) == [
            "num_accepted",
            "next_tokens",
            "next_draft_tokens",
        ]
        sampled = spec_decode_outputs([*base, probs])
        assert list(sampled["spec"]) == [
            "num_accepted",
            "next_tokens",
            "next_draft_tokens",
            "next_draft_probs",
        ]
        scored = spec_decode_outputs([*base, scores], scores=True)
        assert list(scored["spec"]) == [
            "num_accepted",
            "next_tokens",
            "next_draft_tokens",
            "next_draft_scores",
        ]
        lengths = rows(DType.int32, (2,))
        per_row = spec_decode_outputs(
            [*base, scores, lengths, probs],
            scores=True,
            verify_lengths=True,
        )
        assert list(per_row["spec"]) == [
            "num_accepted",
            "next_tokens",
            "next_draft_tokens",
            "next_draft_scores",
            "verify_lengths",
            "next_draft_probs",
        ]
        graph.output(*base, scores, lengths, probs)


class _StubDriver:
    """Declares a per-row verify or not, and decodes inputs as they come."""

    def __init__(self, per_row: bool) -> None:
        self.input_spec = SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU()],
            distributed=False,
            include_per_row_verify=per_row,
            num_draft_positions=K if per_row else None,
        )

    def decode_inputs(self, values: object, kv_params: object) -> object:
        return values


@dataclass(frozen=True)
class _StubCall(_DriverCall):
    """Returns ``count`` copies of the graph's input as the driver's outputs."""

    count: int = 3

    def __call__(
        self,
        graph_inputs: Any,
        signal_buffers: Sequence[BufferValue] | None = None,
    ) -> Sequence[TensorValue]:
        return [graph_inputs[0]] * self.count


def _module_output_keys(per_row: bool, count: int) -> list[str]:
    call = _StubCall(
        driver=cast("Any", _StubDriver(per_row)),
        target_arch="",
        method="",
        target_config=None,
        kv_params=cast("Any", None),
        primary="",
        penalties=(),
        input_types=(),
        boundary_count=0,
        export_contract=None,
        count=count,
    )
    module = _SpecDecodeModule(
        ModuleList(), ModuleList(), lambda target, draft: call
    )
    staged = module.trace(TensorType(DType.int64, [2], DeviceRef.CPU()))
    children = treedef_to_json(staged.out_tree)["children"]
    assert isinstance(children, list)
    (spec,) = children
    assert isinstance(spec, dict)
    return list(spec["keys"])


@pytest.mark.parametrize(
    "per_row, count, keys",
    [
        (False, 3, []),
        (False, 4, ["next_draft_probs"]),
        (True, 5, ["next_draft_scores", "verify_lengths"]),
        (True, 6, ["next_draft_scores", "verify_lengths", "next_draft_probs"]),
    ],
)
def test_the_fused_module_names_the_per_row_verify_outputs(
    per_row: bool, count: int, keys: list[str]
) -> None:
    """The ModuleV3 fused module names the scores and verify lengths as the
    graph API's fused graph does, so an export never reads them as draft
    distributions."""
    assert _module_output_keys(per_row, count) == [
        "num_accepted",
        "next_tokens",
        "next_draft_tokens",
        *keys,
    ]
