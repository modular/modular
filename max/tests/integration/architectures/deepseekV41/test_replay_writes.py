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
"""Checks submitted DSpark write coverage with host token buffers."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41.model import (
    DeepseekV41BatchProcessor,
    DeepseekV41Inputs,
)
from max.pipelines.architectures.deepseekV41.replay import (
    ReplayPlan,
    build_replay_plan,
)
from max.pipelines.architectures.deepseekV41.replay_writes import (
    DSparkWriteBatch,
    DSparkWriteWitness,
)
from max.pipelines.context import TextContext
from max.pipelines.context.status import GenerationStatus
from max.pipelines.context.tokens import TokenBuffer

LEAF_ID = "swa.sliding_window_group(128)"


def prefill(length: int = 512) -> TextContext:
    return TextContext(
        max_length=100000,
        tokens=TokenBuffer(np.arange(length, dtype=np.int64)),
    )


def decode(start: int = 512) -> TextContext:
    ctx = prefill(start)
    ctx.advance_token_buffer(1)
    return ctx


def advance(ctx: TextContext, count: int) -> None:
    for _ in range(count):
        ctx.advance_token_buffer(1)


def witness(
    ctx: TextContext, end: int = 512, width: int = 4
) -> DSparkWriteWitness:
    return DSparkWriteWitness(ctx, end, width, ctx.request_id)


def test_first_trim_publishes_selected_suffix_only_after_finalize() -> None:
    ctx = prefill()
    plan = build_replay_plan([ctx], window_size=128, min_tokens=256)
    assert plan is not None
    batch = DSparkWriteBatch.capture(
        [ctx], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    assert ctx.kv_cache_write_ranges == {}
    advance(ctx, 1)
    previous = batch.publish([ctx])
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (384, 512)}
    assert previous == (witness(ctx, width=0),)


def test_draft_less_mixed_partial_step_publishes_decode_rows_exactly() -> None:
    prompt, row = prefill(), decode()
    plan = build_replay_plan([prompt, row], window_size=128, min_tokens=256)
    assert plan is not None
    batch = DSparkWriteBatch.capture(
        [prompt, row], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    advance(prompt, 1)
    advance(row, 1)
    witnesses = batch.publish([prompt, row])
    assert prompt.kv_cache_write_ranges == {LEAF_ID: (384, 512)}
    assert row.kv_cache_write_ranges == {LEAF_ID: (512, 513)}
    assert witnesses == (witness(prompt, width=0), witness(row, 513, 0))


def test_mixed_partial_step_chains_into_a_verify_step_without_a_gap() -> None:
    prompt, row = prefill(), decode()
    plan = build_replay_plan([prompt, row], window_size=128, min_tokens=256)
    assert plan is not None
    first = DSparkWriteBatch.capture(
        [prompt, row], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    advance(prompt, 1)
    advance(row, 1)
    witnesses = first.publish([prompt, row])
    first_range = row.kv_cache_write_ranges[LEAF_ID]
    second = DSparkWriteBatch.capture(
        [row], None, leaf_id=LEAF_ID, verified_width=3, previous=witnesses
    )
    advance(row, 1)
    second.publish([row])
    assert row.kv_cache_write_ranges[LEAF_ID] == (first_range[1], 514)


def test_in_flight_row_in_a_partial_batch_gets_no_coverage() -> None:
    prompt, row = prefill(), decode()
    row.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    plan = ReplayPlan(
        kept_indices=np.array([], dtype=np.int64),
        kept_offsets=np.array([0], dtype=np.uint32),
        # G1: an in-flight row is never trimmed, so its skip is 0.
        kept_skip=np.array([384, 0], dtype=np.int32),
        verify_width=0,
    )
    batch = DSparkWriteBatch.capture(
        [prompt, row], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    advance(prompt, 1)
    advance(row, 2)
    witnesses = batch.publish([prompt, row])
    start, end = row.kv_cache_write_ranges[LEAF_ID]
    assert start == end
    assert witnesses == (witness(prompt, width=0),)


@pytest.mark.parametrize("accepted", [0, 1, 4])
def test_overlap_unions_prior_accepted_prefix_and_shifted_current_write(
    accepted: int,
) -> None:
    ctx = decode()
    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=3, previous=(witness(ctx),)
    )
    advance(ctx, accepted + 1)
    previous = batch.publish([ctx])
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (512, 513 + accepted)}
    assert previous == (witness(ctx, end=513 + accepted, width=3),)


@pytest.mark.parametrize("accepted", [0, 1, 4])
def test_early_sync_joins_already_realized_accepted_prefix(
    accepted: int,
) -> None:
    ctx = decode()
    prior = witness(ctx)
    advance(ctx, accepted)
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=2, previous=(prior,)
    )
    advance(ctx, 1)
    batch.publish([ctx])
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (512, 513 + accepted)}


@pytest.mark.parametrize(
    "base_end, prior_width, optimistic, advance_count",
    [
        (513, 4, 4, 1),
        (512, 3, 4, 1),
        (512, 4, 4, 0),
        (512, 4, 4, 6),
    ],
)
def test_unsupported_rebase_omits_coverage_and_drops_witness(
    base_end: int, prior_width: int, optimistic: int, advance_count: int
) -> None:
    ctx = decode()
    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * optimistic
    batch = DSparkWriteBatch.capture(
        [ctx],
        None,
        leaf_id=LEAF_ID,
        verified_width=4,
        previous=(witness(ctx, end=base_end, width=prior_width),),
    )
    advance(ctx, advance_count)
    assert batch.publish([ctx]) == ()
    assert ctx.kv_cache_write_ranges == {}


@pytest.mark.parametrize(
    "status",
    [GenerationStatus.MAXIMUM_LENGTH, GenerationStatus.END_OF_SEQUENCE],
)
def test_terminal_or_clipped_context_omits_new_coverage(
    status: GenerationStatus,
) -> None:
    ctx = decode()
    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=4, previous=(witness(ctx),)
    )
    advance(ctx, 2)
    ctx.status = status
    assert batch.publish([ctx]) == ()
    assert ctx.kv_cache_write_ranges == {}


def test_missing_or_replaced_previous_context_cannot_certify_accepted_bridge() -> (
    None
):
    ctx = decode()
    other = decode()
    other.request_id = ctx.request_id
    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    batch = DSparkWriteBatch.capture(
        [ctx],
        None,
        leaf_id=LEAF_ID,
        verified_width=4,
        previous=(witness(other),),
    )
    advance(ctx, 2)
    assert batch.publish([ctx]) == ()
    assert ctx.kv_cache_write_ranges == {}


def test_changed_request_identity_cannot_reuse_previous_witness() -> None:
    ctx = decode()
    prior = replace(witness(ctx), request_id=decode().request_id)
    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=4, previous=(prior,)
    )
    advance(ctx, 2)
    assert batch.publish([ctx]) == ()
    assert ctx.kv_cache_write_ranges == {}


def test_reset_drops_conditional_prefix_and_certifies_new_full_prompt() -> None:
    ctx = decode()
    prior = witness(ctx)
    ctx.reset()
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=0, previous=(prior,)
    )
    advance(ctx, 1)
    batch.publish([ctx])
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (0, 513)}


@pytest.mark.parametrize("replace_context", [False, True])
def test_first_trim_mismatch_latches_empty_coverage(
    replace_context: bool,
) -> None:
    ctx = prefill()
    plan = build_replay_plan([ctx], window_size=128, min_tokens=256)
    assert plan is not None
    batch = DSparkWriteBatch.capture(
        [ctx], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    actual = prefill() if replace_context else ctx
    assert batch.publish([actual]) == ()
    assert actual.kv_cache_write_ranges == {LEAF_ID: (0, 0)}


def test_eager_full_write_can_restart_after_an_uncertified_gap() -> None:
    ctx = decode(520)
    batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=4, previous=()
    )
    advance(ctx, 1)
    assert batch.publish([ctx]) == (witness(ctx, end=521),)
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (520, 521)}


def test_rebatch_uses_context_identity_and_drops_absent_witnesses() -> None:
    first, second, absent = decode(), decode(700), decode(900)
    previous = (
        witness(first),
        witness(second, end=700),
        witness(absent, end=900),
    )
    for ctx in (first, second):
        ctx.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 4
    batch = DSparkWriteBatch.capture(
        [second, first],
        None,
        leaf_id=LEAF_ID,
        verified_width=4,
        previous=previous,
    )
    advance(first, 2)
    advance(second, 4)
    renewed = batch.publish([second, first])
    assert first.kv_cache_write_ranges == {LEAF_ID: (512, 514)}
    assert second.kv_cache_write_ranges == {LEAF_ID: (700, 704)}
    assert tuple(item.context for item in renewed) == (second, first)
    assert absent.kv_cache_write_ranges == {}


def test_terminal_first_trim_latches_empty_coverage() -> None:
    ctx = prefill()
    plan = build_replay_plan([ctx], window_size=128, min_tokens=256)
    assert plan is not None
    batch = DSparkWriteBatch.capture(
        [ctx], plan, leaf_id=LEAF_ID, verified_width=0, previous=()
    )
    advance(ctx, 1)
    ctx.status = GenerationStatus.END_OF_SEQUENCE
    assert batch.publish([ctx]) == ()
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (512, 512)}


def captured_inputs(
    ctx: TextContext, width: int = 2
) -> tuple[DeepseekV41BatchProcessor, DeepseekV41Inputs]:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs = object.__new__(DeepseekV41Inputs)
    inputs.replay_write_batch = DSparkWriteBatch.capture(
        [ctx], None, leaf_id=LEAF_ID, verified_width=width, previous=()
    )
    return processor, inputs


def test_captured_full_associates_only_after_selected_success() -> None:
    ctx = prefill()
    processor, inputs = captured_inputs(ctx)
    snapshot = inputs.replay_write_batch
    assert snapshot is not None
    processor._submitted_replay_writes = snapshot
    processor.begin_captured_submission(inputs)
    assert processor._submitted_replay_writes is None
    assert processor._previous_replay_writes == ()
    assert ctx.kv_cache_write_ranges == {}
    processor.on_captured_submission_success(inputs, 2)
    assert processor._submitted_replay_writes is snapshot
    advance(ctx, 1)
    processor.finalize_kv_cache_writes([ctx])
    assert ctx.kv_cache_write_ranges == {LEAF_ID: (0, 512)}
    assert processor._submitted_replay_writes is None


@pytest.mark.parametrize("partial_kind", ["plan", "buffers"])
def test_capture_rejects_partial_before_selected_submission(
    partial_kind: str,
) -> None:
    ctx = prefill()
    processor, inputs = captured_inputs(ctx)
    processor._submitted_replay_writes = inputs.replay_write_batch
    if partial_kind == "plan":
        inputs.replay_plan = build_replay_plan(
            [ctx], window_size=128, min_tokens=256
        )
        assert inputs.replay_plan is not None
    else:
        inputs.replay_plan = None
        inputs.replay_buffers = (Mock(),)
    with pytest.raises(RuntimeError, match="require FULL"):
        processor.begin_captured_submission(inputs)
    assert processor._submitted_replay_writes is None
    assert ctx.kv_cache_write_ranges == {}


@pytest.mark.parametrize("actual_width", [1, 3])
def test_capture_width_mismatch_cannot_associate(actual_width: int) -> None:
    ctx = prefill()
    processor, inputs = captured_inputs(ctx)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    processor.begin_captured_submission(inputs)
    with pytest.raises(RuntimeError, match="verify width"):
        processor.on_captured_submission_success(inputs, actual_width)
    assert processor._submitted_replay_writes is None
    assert ctx.kv_cache_write_ranges == {}


@pytest.mark.parametrize(
    "invalid", ["terminal", "replacement", "request_id", "reordered"]
)
def test_captured_full_finalize_retains_identity_and_terminal_guards(
    invalid: str,
) -> None:
    ctx = prefill()
    processor, inputs = captured_inputs(ctx)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    processor.begin_captured_submission(inputs)
    processor.on_captured_submission_success(inputs, 2)
    advance(ctx, 1)
    contexts = [ctx]
    if invalid == "terminal":
        ctx.status = GenerationStatus.END_OF_SEQUENCE
    elif invalid == "replacement":
        other = prefill()
        other.request_id = ctx.request_id
        advance(other, 1)
        contexts = [other]
    elif invalid == "request_id":
        ctx.request_id = prefill().request_id
    else:
        contexts = [prefill(), ctx]
    processor.finalize_kv_cache_writes(contexts)
    assert all(item.kv_cache_write_ranges == {} for item in contexts)
    assert processor._previous_replay_writes == ()


def test_missing_captured_recorder_cannot_supply_new_coverage() -> None:
    ctx = decode()
    processor, inputs = captured_inputs(ctx)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    processor._previous_replay_writes = (witness(ctx),)
    inputs.replay_write_batch = None
    processor.begin_captured_submission(inputs)
    processor.on_captured_submission_success(inputs, 2)
    advance(ctx, 1)
    processor.finalize_kv_cache_writes([ctx])
    assert ctx.kv_cache_write_ranges == {}
    assert processor._previous_replay_writes == ()
