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
"""Checks host suffix planning and replay selection exemptions."""

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41.replay import (
    build_replay_plan,
    build_replay_plan_with_reason,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.pipeline_variants import (
    text_generation as text_pipeline_module,
)
from max.pipelines.modeling.types import TextGenerationInputs


def context(start: int, length: int) -> TextContext:
    tokens = TokenBuffer(np.arange(start + length, dtype=np.int64))
    tokens.skip_processing(start)
    return TextContext(max_length=100000, tokens=tokens)


@pytest.mark.parametrize("end", [256, 257, 383, 384, 4096, 4097])
def test_suffix_alignment(end: int) -> None:
    ctx = context(0, end)
    plan = build_replay_plan([ctx], window_size=128, min_tokens=256)
    assert plan is not None
    expected_start = end // 128 * 128 - 128
    np.testing.assert_array_equal(
        plan.kept_indices, np.arange(expected_start, end)
    )
    assert plan.kept_indices.dtype == np.int64
    assert plan.kept_offsets.dtype == np.uint32
    assert plan.kept_skip.dtype == np.int32
    assert plan.kept_skip.tolist() == [expected_start]
    assert 128 <= len(plan.kept_indices) <= 255
    assert ctx.kv_cache_write_ranges == {}


def test_short_chunk_keeps_previous_tail_and_global_indices() -> None:
    plan = build_replay_plan(
        [context(1024, 4096), context(4096, 3), context(256, 1)],
        window_size=128,
        min_tokens=256,
    )
    assert plan is not None
    assert plan.kept_skip.tolist() == [3968, 0, 0]
    assert plan.kept_offsets.tolist() == [0, 128, 131, 132]
    np.testing.assert_array_equal(
        plan.kept_indices[-4:], [4096, 4097, 4098, 4099]
    )


@pytest.mark.parametrize(
    ("boundaries", "expected"),
    [
        ([0, 3], [128]),
        ([0, 1, 3], [128, 3]),
        ([0, 0, 3], [0, 128]),
        ([0, 3, 3], [128, 0]),
    ],
)
def test_kept_longest_per_replica_including_empty(
    boundaries: list[int], expected: list[int]
) -> None:
    plan = build_replay_plan(
        [context(1024, 4096), context(4096, 3), context(256, 1)],
        window_size=128,
        min_tokens=256,
    )
    assert plan is not None
    longest = plan.kept_longest(boundaries)
    assert longest.dtype == np.uint32
    assert longest.tolist() == expected


@pytest.mark.parametrize("length", [1, 127, 128, 255])
def test_no_reduction_uses_full_graph(length: int) -> None:
    assert (
        build_replay_plan([context(0, length)], window_size=128, min_tokens=1)
        is None
    )


def test_default_off_and_threshold() -> None:
    ctx = context(0, 4096)
    assert build_replay_plan([ctx], window_size=128) is None
    assert build_replay_plan([ctx], window_size=128, min_tokens=8192) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_hidden_states": True},
        {"enable_echo": True},
    ],
)
def test_exemptions(kwargs: dict[str, int | bool]) -> None:
    ctx = context(0, 4096)
    assert (
        build_replay_plan(
            [ctx],
            window_size=128,
            min_tokens=256,
            return_hidden_states=bool(
                kwargs.get("return_hidden_states", False)
            ),
            enable_echo=bool(kwargs.get("enable_echo", False)),
        )
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"min_tokens": 0}, "disabled"),
        ({"return_hidden_states": True}, "hidden_states"),
        ({"enable_echo": True}, "echo"),
        ({"min_tokens": 8192}, "below_threshold"),
    ],
)
def test_skip_reasons(kwargs: dict[str, Any], reason: str) -> None:
    args: dict[str, Any] = {"min_tokens": 256, **kwargs}
    plan, actual = build_replay_plan_with_reason(
        [context(0, 4096)], window_size=128, **args
    )
    assert plan is None
    assert actual == reason


def decoding(start: int, length: int) -> TextContext:
    ctx = context(start, length)
    ctx._is_initial_prompt = False
    return ctx


def test_mixed_step_trims_only_the_prompt_and_keeps_decode_rows_whole() -> None:
    plan = build_replay_plan(
        [context(0, 300), decoding(4096, 1), decoding(4096, 1)],
        window_size=128,
        min_tokens=256,
    )
    assert plan is not None
    expected = [*range(128, 300), 300, 301]
    assert plan.kept_indices.tolist() == expected
    assert plan.kept_offsets.tolist() == [0, 172, 173, 174]
    assert plan.kept_skip.tolist() == [128, 0, 0]
    assert plan.kept_longest([0, 3]).tolist() == [172]


def test_decode_row_after_a_long_prompt_is_kept_whole() -> None:
    plan = build_replay_plan(
        [context(0, 4096), decoding(5000, 1)],
        window_size=128,
        min_tokens=256,
    )
    assert plan is not None
    assert plan.kept_skip.tolist() == [3968, 0]
    assert plan.kept_offsets.tolist() == [0, 128, 129]


def test_mixed_step_threshold_uses_the_longest_active_row() -> None:
    def reason(min_tokens: int) -> str | None:
        return build_replay_plan_with_reason(
            [context(0, 300), decoding(4096, 1)],
            window_size=128,
            min_tokens=min_tokens,
        )[1]

    assert reason(300) is None
    assert reason(301) == "below_threshold"


def test_verify_width_plans_in_the_merged_row_space() -> None:
    plan = build_replay_plan(
        [context(0, 300), decoding(4096, 1)],
        window_size=128,
        min_tokens=256,
        verify_width=3,
    )
    assert plan is not None
    # Each request owns its prompt rows plus 3 draft rows, drafts last.
    expected = [*range(128, 303), *range(303, 307)]
    assert plan.kept_indices.tolist() == expected
    assert plan.kept_offsets.tolist() == [0, 175, 179]
    assert plan.kept_skip.tolist() == [128, 0]
    assert plan.kept_longest([0, 2]).tolist() == [175]


def test_verify_width_offsets_accumulate_across_prompt_rows() -> None:
    plan = build_replay_plan(
        [context(0, 300), context(256, 400), decoding(512, 1)],
        window_size=128,
        min_tokens=256,
        verify_width=2,
    )
    assert plan is not None
    # Merged rows: [0, 302), [302, 704), [704, 707).
    assert plan.kept_skip.tolist() == [128, 256, 0]
    expected = [
        *range(128, 302),
        *range(302 + 256, 704),
        *range(704, 707),
    ]
    assert plan.kept_indices.tolist() == expected
    assert plan.kept_offsets.tolist() == [0, 174, 320, 323]


def test_zero_verify_width_is_the_prompt_space() -> None:
    contexts = [context(0, 300), decoding(4096, 1)]
    explicit = build_replay_plan(
        contexts, window_size=128, min_tokens=256, verify_width=0
    )
    default = build_replay_plan(contexts, window_size=128, min_tokens=256)
    assert explicit is not None and default is not None
    np.testing.assert_array_equal(explicit.kept_indices, default.kept_indices)
    np.testing.assert_array_equal(explicit.kept_skip, default.kept_skip)


def test_negative_verify_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        build_replay_plan(
            [context(0, 4096)],
            window_size=128,
            min_tokens=256,
            verify_width=-1,
        )


def test_in_flight_row_is_kept_whole_without_refusing_the_step() -> None:
    # G1: whatever the row's host length, a row with in-flight drafts never
    # skips, so a skipped row always has device cache_lengths equal to
    # processed_length.
    width = 4
    row = context(4096, 4096)
    row.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * width
    plan = build_replay_plan(
        [context(0, 300), row, decoding(8192, 1)],
        window_size=128,
        min_tokens=256,
        verify_width=width,
    )
    assert plan is not None
    assert plan.kept_skip.tolist() == [128, 0, 0]
    sizes = np.diff(plan.kept_offsets).tolist()
    assert sizes == [300 + width - 128, 4096 + width, 1 + width]


def test_in_flight_rows_alone_have_nothing_to_remove() -> None:
    row = context(4096, 4096)
    row.spec_decoding_state.maybe_accepted_draft_tokens = [-1] * 2
    plan, reason = build_replay_plan_with_reason(
        [row], window_size=128, min_tokens=256, verify_width=2
    )
    assert plan is None
    assert reason == "nothing_to_remove"


def test_skip_reasons_for_batch_contents() -> None:
    def reason(contexts: list[TextContext]) -> str | None:
        return build_replay_plan_with_reason(
            contexts, window_size=128, min_tokens=256
        )[1]

    assert reason([]) == "no_contexts"
    assert reason([context(0, 200)]) == "below_threshold"
    no_prompt = context(0, 4096)
    no_prompt._is_initial_prompt = False
    assert reason([no_prompt]) == "no_initial_prompt"
    echoing = context(0, 4096)
    echoing.log_probabilities_echo = True
    assert reason([echoing]) == "logprob_echo"
    assert reason([context(0, 4096)]) is None


def test_request_echo_and_decode_exemptions() -> None:
    ctx = context(0, 4096)
    ctx.log_probabilities_echo = True
    assert build_replay_plan([ctx], window_size=128, min_tokens=256) is None
    ctx.log_probabilities_echo = False
    ctx._is_initial_prompt = False
    assert build_replay_plan([ctx], window_size=128, min_tokens=256) is None


def test_empty_batch_and_invalid_window() -> None:
    assert build_replay_plan([], window_size=128, min_tokens=256) is None
    with pytest.raises(ValueError, match="positive"):
        build_replay_plan([], window_size=0)


def test_launch_failure_skips_metadata_and_cache_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context(0, 4096)
    pipeline = object.__new__(text_pipeline_module.TextGenerationPipeline)
    model_inputs = Mock()
    monkeypatch.setattr(
        pipeline,
        "prepare_batch",
        Mock(return_value=(model_inputs, None, [ctx])),
    )
    pipeline._pipeline_config = Mock()
    pipeline._sampler_without_bitmask = Mock()
    pipeline._sampler_device = Mock()
    pipeline._pinned_new_tokens = Mock()
    pipeline._identity_logit_offsets = Mock()
    pipeline.vocab_size = 128
    pipeline_model = Mock(
        execute=Mock(side_effect=RuntimeError("launch failed"))
    )
    kv_manager = Mock()
    pipeline._pipeline_model = pipeline_model
    pipeline._kv_manager = kv_manager
    monkeypatch.setattr(text_pipeline_module, "FusedSamplingProcessor", Mock())
    inputs = TextGenerationInputs(batches=[[ctx]])
    with pytest.raises(RuntimeError, match="launch failed"):
        pipeline._execute(inputs)
    pipeline_model.finalize_kv_cache_writes.assert_not_called()
    kv_manager.step.assert_not_called()
    assert ctx.kv_cache_write_ranges == {}
