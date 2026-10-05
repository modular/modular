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
"""How many carried drafts a step verifies, and the host mirror of that trim.

The width is a pure function of the batch, because two different actors shape
buffers from it -- the execute path builds the draft array, and the async
FSM callback fills the constrained-decoding bitmask -- and they have to agree
without communicating.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import click
import numpy as np
import pytest
from click.testing import CliRunner
from max._entrypoints.cli.config import config_to_flag
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.lib import PipelineArgs
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
    _host_mirror_realized_drafts,
    _mixed_verify_width,
    _reachable_verify_widths,
    _verify_width_candidates,
    _verify_widths_by_batch_size,
)
from max.pipelines.modeling.types.pipeline_variants.text_generation import (
    BatchType,
    CompletedBatchStats,
)
from max.pipelines.speculative.adaptive_width import AdaptiveVerifyWidth
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    VerifyWidthRange,
)
from max.pipelines.speculative.utils import _SpeculativeDecodingMetrics


@dataclass
class _Tokens:
    generated_length: int


@dataclass
class _Ctx:
    tokens: _Tokens
    # ``_should_verify_drafts`` reads these directly, so a fake without them
    # raises instead of reporting an unconstrained row.
    matcher: object | None = None
    grammar: object | None = None
    json_schema: object | None = None


@dataclass
class _Inputs:
    batches: list[list[_Ctx]] = field(default_factory=list)

    @property
    def flat_batch(self) -> list[_Ctx]:
        return [ctx for batch in self.batches for ctx in batch]

    @property
    def batch_type(self) -> BatchType:
        """Mirrors ``TextGenerationInputs``: one prefill row makes it CE."""
        return (
            BatchType.CE
            if any(c.tokens.generated_length == 0 for c in self.flat_batch)
            else BatchType.TG
        )


def _decode_batches(*sizes: int) -> _Inputs:
    """Replica batches whose every row has already generated a token."""
    return _Inputs([[_Ctx(_Tokens(1)) for _ in range(n)] for n in sizes])


def _mixed_batch(decode_rows: int, prefill_rows: int = 1) -> _Inputs:
    """One replica batch carrying both decode and freshly prefilled rows."""
    return _Inputs(
        [
            [_Ctx(_Tokens(1)) for _ in range(decode_rows)]
            + [_Ctx(_Tokens(0)) for _ in range(prefill_rows)]
        ]
    )


def _pipeline(
    *,
    configured: int,
    lookup: list[int] | None,
    mixed_width: int | None = None,
    allow_mixed: bool = False,
    adaptive: AdaptiveVerifyWidth | None = None,
) -> OverlapTextGenerationPipeline[Any]:
    pipeline = object.__new__(OverlapTextGenerationPipeline)
    spec_state = type("_S", (), {"num_speculative_tokens": configured})()
    pipeline._spec_decode_state = cast(Any, spec_state)
    pipeline._widths_by_batch_size = (
        [[configured]] if lookup is None else [[width] for width in lookup]
    )
    pipeline._mixed_verify_width = mixed_width
    pipeline._allow_mixed_verify = allow_mixed
    pipeline._adaptive_width = adaptive
    return pipeline


def _width(
    inputs: _Inputs,
    *,
    configured: int,
    lookup: list[int] | None,
    mixed_width: int | None = None,
    allow_mixed: bool = False,
    adaptive: AdaptiveVerifyWidth | None = None,
) -> int:
    pipeline = _pipeline(
        configured=configured,
        lookup=lookup,
        mixed_width=mixed_width,
        allow_mixed=allow_mixed,
        adaptive=adaptive,
    )
    return OverlapTextGenerationPipeline._verify_width(
        pipeline, cast(Any, inputs)
    )


def test_no_schedule_verifies_every_carried_draft() -> None:
    assert _width(_decode_batches(8), configured=3, lookup=None) == 3


def test_schedule_selects_by_batch_size() -> None:
    # batch_size -> width; index 0 unused.
    lookup = [0, 3, 3, 3, 1, 1]
    assert _width(_decode_batches(2), configured=3, lookup=lookup) == 3
    assert _width(_decode_batches(5), configured=3, lookup=lookup) == 1
    # A batch size past the lookup tail saturates rather than indexing off it.
    assert _width(_decode_batches(999), configured=3, lookup=lookup) == 1


def test_a_row_with_no_generated_token_collapses_the_width() -> None:
    """A freshly prefilled row carries no proposals, so nothing is verified."""
    inputs = _Inputs([[_Ctx(_Tokens(1)), _Ctx(_Tokens(0))]])
    assert _width(inputs, configured=3, lookup=None) == 0


def test_empty_batch_does_not_index_off_the_lookup() -> None:
    assert _width(_Inputs([]), configured=3, lookup=[0, 3, 1]) == 3


def test_width_is_the_per_replica_maximum_not_the_total() -> None:
    """Under DP the schedule is indexed by per-replica decode batch size.

    Eight requests spread over four replicas is a per-replica batch of 2, which
    is a small batch on every GPU -- not a large one.
    """
    lookup = [0, 3, 3, 3, 1, 1, 1, 1, 1]
    inputs = _decode_batches(2, 2, 2, 2)
    assert len(inputs.flat_batch) == 8
    assert _width(inputs, configured=3, lookup=lookup) == 3


# ---------------------------------------------------------------------------
# narrowing a mixed prefill+decode step
# ---------------------------------------------------------------------------
#
# A mixed step's width is on the critical path of every prompt in it, and
# unlike a pure decode step it runs eager -- graph replay is TG-only, and one
# prefill row makes the batch CE -- so narrowing it costs no captured graph.


def test_mixed_batch_takes_the_mixed_width() -> None:
    assert (
        _width(
            _mixed_batch(8),
            configured=7,
            lookup=None,
            mixed_width=3,
            allow_mixed=True,
        )
        == 3
    )


def test_mixed_width_leaves_pure_decode_at_full_depth() -> None:
    """The point of the knob: only the prefill-carrying steps narrow."""
    assert (
        _width(
            _decode_batches(9),
            configured=7,
            lookup=None,
            mixed_width=3,
            allow_mixed=True,
        )
        == 7
    )


def test_mixed_width_overrides_the_batch_size_schedule() -> None:
    """The two narrowings answer different questions, so mixed wins on a
    mixed step regardless of what the batch size would have selected."""
    lookup = [0] + [7] * 4 + [5] * 5
    assert (
        _width(
            _mixed_batch(8),
            configured=7,
            lookup=lookup,
            mixed_width=3,
            allow_mixed=True,
        )
        == 3
    )
    assert (
        _width(
            _decode_batches(9),
            configured=7,
            lookup=lookup,
            mixed_width=3,
            allow_mixed=True,
        )
        == 5
    )


def test_unset_mixed_width_leaves_mixed_batches_on_the_schedule() -> None:
    lookup = [0] + [7] * 4 + [5] * 5
    assert (
        _width(_mixed_batch(8), configured=7, lookup=lookup, allow_mixed=True)
        == 5
    )


def test_mixed_width_does_not_resurrect_a_batch_that_cannot_verify() -> None:
    """The width knob sits behind the verify gate, not in front of it."""
    # Mixed verification switched off: the batch verifies nothing at all.
    assert (
        _width(
            _mixed_batch(8),
            configured=7,
            lookup=None,
            mixed_width=3,
            allow_mixed=False,
        )
        == 0
    )
    # Pure prefill carries no proposals no matter what is configured.
    pure_prefill = _Inputs([[_Ctx(_Tokens(0)) for _ in range(4)]])
    assert (
        _width(
            pure_prefill,
            configured=7,
            lookup=None,
            mixed_width=3,
            allow_mixed=True,
        )
        == 0
    )


def test_mixed_width_is_capped_at_the_configured_depth() -> None:
    """``dflash`` resolves its depth from the checkpoint after config
    validation, so the ceiling is applied here rather than by the config."""
    assert _mixed_verify_width(_config(None, mixed_width=9), 3) == 3
    assert _mixed_verify_width(_config(None, mixed_width=1), 3) == 1


def test_reachable_widths_include_the_mixed_width() -> None:
    """The constrained-decoding bitmask allocates one pinned buffer per
    reachable width. A mixed width absent from this set killed the model worker
    with "prime() num_positions 2 has no allocated buffer" on the first mixed
    batch, rather than degrading, so the set has to cover it.
    """
    # No schedule: the only batch-size width is the full depth.
    assert _reachable_verify_widths(_config(None, mixed_width=1), 3, 8) == [
        1,
        3,
    ]
    # With a schedule, the mixed width joins the scheduled widths.
    scheduled = _config([(1, 2, 3), (3, 8, 2)], mixed_width=1)
    assert _reachable_verify_widths(scheduled, 3, 8) == [1, 2, 3]
    # A mixed width that clamps to the full depth adds nothing new.
    assert _reachable_verify_widths(_config(None, mixed_width=9), 3, 8) == [3]
    # Unset leaves the reachable set exactly as it was.
    assert _reachable_verify_widths(_config(None), 3, 8) == [3]


def test_no_mixed_width_configured_is_none() -> None:
    assert _mixed_verify_width(_config(None), 3) is None
    # Set, but on a pipeline that does not speculate at all.
    assert _mixed_verify_width(_config(None, mixed_width=3), 0) is None


# ---------------------------------------------------------------------------
# the schedule as it arrives from SpeculativeConfig
# ---------------------------------------------------------------------------


def _config(
    schedule: list[tuple[int, int, int]] | None,
    *,
    method: str = "eagle",
    mixed_width: int | None = None,
    widths: str | None = None,
    min_batch_size: int = 1,
    configured: int | None = 3,
) -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method=cast(Any, method),
        num_speculative_tokens=configured,
        num_speculative_tokens_mixed_batch=mixed_width,
        adaptive_speculative_widths=widths,
        adaptive_speculative_min_batch_size=min_batch_size,
        num_speculative_tokens_per_batch_size=(
            None
            if schedule is None
            else [
                VerifyWidthRange(
                    batch_start=start, batch_end=end, num_tokens=count
                )
                for start, end, count in schedule
            ]
        ),
    )


def test_no_schedule_leaves_the_width_at_the_configured_depth() -> None:
    assert _verify_widths_by_batch_size(_config(None), 3, 2) == [[3]] * 3
    # A schedule set on a pipeline that does not speculate at all.
    assert _verify_widths_by_batch_size(_config([(1, 8, 1)]), 0, 2) == [[0]] * 3


def test_schedule_builds_a_dense_lookup() -> None:
    config = _config([(1, 2, 3), (3, 8, 1)])
    assert _verify_widths_by_batch_size(config, 3, 4) == [
        [0],
        [3],
        [3],
        [1],
        [1],
    ]


def test_block_drafters_apply_the_schedule_too() -> None:
    """A block drafter's block width is baked into its checkpoint, but the
    target still only verifies a prefix of that block per the schedule --
    the draft itself always produces the full block regardless.
    """
    config = _config([(1, 2, 3), (3, 8, 1)], method="dflash")
    assert config.num_speculative_tokens_per_batch_size is not None
    assert _verify_widths_by_batch_size(config, 3, 4) == [
        [0],
        [3],
        [3],
        [1],
        [1],
    ]


def test_config_rejects_a_schedule_with_a_gap_at_the_front() -> None:
    with pytest.raises(ValueError, match="must start at batch size 1"):
        _config([(2, 8, 1)])


# ---------------------------------------------------------------------------
# the host mirror of the device realize-scatter
# ---------------------------------------------------------------------------
#
# The mirror has to trim the previous step's proposals to this step's verify
# width exactly as the device graph does. The previous step always drafted the
# configured depth, so once a step verifies fewer than it drafted the two
# arrays stop having the same width -- and mirroring untrimmed raised
# "could not broadcast input array from shape (3,) into shape (1,)".

_MAGIC = -7


def test_mirror_trims_to_a_narrower_verify_width() -> None:
    """Drafted 3, verifying 1: the tail is dropped, not an error."""
    realized = _host_mirror_realized_drafts(
        np.full((2, 1), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    np.testing.assert_array_equal(realized, np.array([[11], [21]]))


def test_mirror_leaves_equal_widths_unchanged() -> None:
    prev_next = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64)
    realized = _host_mirror_realized_drafts(
        np.full((2, 3), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        prev_next,
    )
    np.testing.assert_array_equal(realized, prev_next)


def test_mirror_of_a_zero_verify_width_is_an_empty_array() -> None:
    """A prefill->decode boundary step verifies nothing."""
    realized = _host_mirror_realized_drafts(
        np.zeros((2, 0), dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    assert realized.shape == (2, 0)


def _prepare_synthesis(
    *,
    configured: int,
    schedule: list[tuple[int, int, int]] | None = None,
    widths: str | None = None,
    max_batch: int = 4,
) -> OverlapTextGenerationPipeline[Any]:
    """Runs ``prepare_graph_synthesis_buckets`` against a real aligner."""
    config = _config(schedule, widths=widths, configured=configured)
    pipeline = object.__new__(OverlapTextGenerationPipeline)
    pipeline._spec_decode_state = cast(
        Any, type("_S", (), {"num_speculative_tokens": configured})()
    )
    pipeline._pipeline_config = cast(
        Any, type("_P", (), {"speculative": config})()
    )
    pipeline._max_batch_size = max_batch
    pipeline._verify_widths = _verify_width_candidates(
        config, configured, max_batch
    )
    pipeline._kv_manager = cast(
        Any,
        type(
            "_KV",
            (),
            {
                "params": MHAKVCacheParams(
                    dtype=DType.bfloat16,
                    head_dim=64,
                    num_layers=2,
                    devices=[DeviceRef.GPU()],
                    n_kv_heads=8,
                ),
                "effective_max_seq_length": None,
            },
        )(),
    )
    pipeline._pipeline_model = cast(
        Any, type("_M", (), {"max_seq_len": 2048})()
    )
    OverlapTextGenerationPipeline.prepare_graph_synthesis_buckets(pipeline)
    return pipeline


def test_synthesis_names_every_unservable_width() -> None:
    with pytest.raises(
        ValueError, match=r"a step verifying \[1, 2\] cannot be served"
    ):
        _prepare_synthesis(
            configured=3, schedule=[(1, 1, 3), (2, 2, 2), (3, 4, 1)]
        )


def test_synthesis_refuses_adaptive_widths() -> None:
    with pytest.raises(
        ValueError, match=r"a step verifying \[1, 2\] cannot be served"
    ):
        _prepare_synthesis(configured=3, widths="all")


@pytest.mark.parametrize(
    "schedule", [[(1, 4, 3)], None], ids=["unnarrowed", "unscheduled"]
)
def test_synthesis_accepts_an_unnarrowed_schedule(
    schedule: list[tuple[int, int, int]] | None,
) -> None:
    pipeline = _prepare_synthesis(configured=3, schedule=schedule)
    assert pipeline._synthesis_aligner is not None


SCHEDULE = [(1, 8, 7), (9, 16, 3), (17, 32, 1)]


def _candidates(
    widths: str | None = None,
    schedule: list[tuple[int, int, int]] | None = None,
    *,
    configured: int = 7,
    max_batch: int = 32,
) -> list[int]:
    return _verify_width_candidates(
        _config(schedule, widths=widths, configured=configured),
        configured,
        max_batch,
    )


@pytest.mark.parametrize(
    ("widths", "expected"), [("1,3,5", [1, 3, 5]), ("5,1,3,1", [1, 3, 5])]
)
def test_candidates_are_the_named_widths(
    widths: str, expected: list[int]
) -> None:
    assert _candidates(widths) == expected


def test_all_expands_to_every_width_the_drafter_carries() -> None:
    assert _candidates("all", configured=4) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("widths", "schedule"),
    [(None, None), ("1,3,5", None), ("all", None), (None, SCHEDULE)],
    ids=["unset", "explicit", "all", "schedule"],
)
def test_bitmask_widths_match_the_candidates(
    widths: str | None, schedule: list[tuple[int, int, int]] | None
) -> None:
    """A width the bitmask lacks fails the step with "no allocated buffer"."""
    config = _config(schedule, widths=widths, configured=7)
    assert _reachable_verify_widths(config, 7, 32) == _verify_width_candidates(
        config, 7, 32
    )


def test_adaptive_widths_do_not_depend_on_the_batch_ceiling() -> None:
    """Graph capture derives widths at a lower batch ceiling than the bitmask."""
    assert _candidates("1,3,7", max_batch=4) == _candidates("1,3,7")


def test_adaptive_table_offers_every_candidate_at_every_batch_size() -> None:
    """The default floor of 1 captures what it did before the floor existed."""
    config = _config(None, widths="1,3,7", configured=7)
    assert _verify_widths_by_batch_size(config, 7, 3)[1:] == [[1, 3, 7]] * 3


def test_adaptive_table_keeps_only_the_widest_below_the_floor() -> None:
    config = _config(None, widths="1,3,7", min_batch_size=3, configured=7)
    assert _verify_widths_by_batch_size(config, 7, 4) == [
        [7],
        [7],
        [7],
        [1, 3, 7],
        [1, 3, 7],
    ]


def test_the_floor_does_not_shrink_the_bitmask_widths() -> None:
    """A narrow width stays reachable at batch sizes above the floor."""
    config = _config(None, widths="1,3,7", min_batch_size=64, configured=7)
    assert _reachable_verify_widths(config, 7, 32) == [1, 3, 7]


def test_a_floor_without_adaptive_widths_is_refused() -> None:
    with pytest.raises(ValueError, match="requires adaptive_speculative"):
        _config(None, min_batch_size=4)


def test_adaptive_widths_and_a_schedule_are_exclusive() -> None:
    with pytest.raises(ValueError, match="set only one"):
        _config(SCHEDULE, widths="1,3")


@pytest.mark.parametrize("widths", ["0", "1,0,3", "-1", "1,9"])
def test_out_of_range_widths_are_refused_at_config_read(widths: str) -> None:
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        _config(None, widths=widths)


def test_dflash_checks_the_ceiling_at_pipeline_build() -> None:
    """dflash reads its ceiling from the checkpoint, after config read."""
    config = _config(None, method="dflash", widths="1,9", configured=None)
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        _verify_width_candidates(config, 3, 32)


@pytest.mark.parametrize("method", ["eagle", "dflash"])
def test_a_malformed_list_is_refused_at_config_read(method: str) -> None:
    configured = None if method == "dflash" else 3
    with pytest.raises(ValueError, match="as a verify width"):
        _config(None, method=method, widths="1,three", configured=configured)
    with pytest.raises(ValueError, match=r"\[0\] are outside 1\.\."):
        _config(None, method=method, widths="0,3", configured=configured)


def _adaptive_pipeline(
    widths: str = "1,3,7",
    *,
    configured: int = 7,
    mixed_width: int | None = None,
    min_batch_size: int = 1,
) -> OverlapTextGenerationPipeline[Any]:
    """A pipeline wired from the flag the way ``__init__`` wires one."""
    config = _config(
        None,
        widths=widths,
        mixed_width=mixed_width,
        min_batch_size=min_batch_size,
        configured=configured,
    )
    pipeline = _pipeline(
        configured=configured,
        lookup=None,
        mixed_width=_mixed_verify_width(config, configured),
        allow_mixed=True,
        adaptive=AdaptiveVerifyWidth(
            _verify_width_candidates(config, configured, 32),
            min_batch_size=min_batch_size,
        ),
    )
    pipeline._widths_by_batch_size = _verify_widths_by_batch_size(
        config, configured, 32
    )
    return pipeline


def _width_of(
    pipeline: OverlapTextGenerationPipeline[Any], inputs: _Inputs
) -> int:
    return OverlapTextGenerationPipeline._verify_width(
        pipeline, cast(Any, inputs)
    )


def _metrics(
    rows: int, width: int, fraction: float
) -> _SpeculativeDecodingMetrics:
    """Metrics a step verifying ``width`` drafts over ``rows`` could report."""
    rows = rows if width else 0
    accepted = min(width, round(width * fraction))
    return _SpeculativeDecodingMetrics(
        num_speculative_tokens=width,
        accepted_per_position=[rows] * accepted + [0] * (width - accepted),
        num_verifications=rows,
    )


@dataclass
class _Batch:
    inputs: _Inputs


def _stats(
    seconds: float, early_sync_duration_s: float | None = None
) -> CompletedBatchStats:
    return CompletedBatchStats(
        batch_type=BatchType.TG,
        batch_size=0,
        num_input_tokens=0,
        num_context_tokens=0,
        execution_time_s=seconds,
        early_sync_duration_s=early_sync_duration_s,
    )


def _record(
    pipeline: OverlapTextGenerationPipeline[Any],
    inputs: _Inputs,
    metrics: _SpeculativeDecodingMetrics,
    stats: CompletedBatchStats | None,
) -> None:
    OverlapTextGenerationPipeline._record_spec_decode_metrics(
        pipeline, cast(Any, _Batch(inputs)), metrics, stats
    )


def _steep(width: int) -> float:
    """A step time that each verified draft raises by half."""
    return 1 + 0.5 * width


def _step(
    pipeline: OverlapTextGenerationPipeline[Any],
    inputs: _Inputs,
    fraction: float,
    step_time: Callable[[int], float] = _steep,
) -> int:
    """Resolves one step's width, then trains on that step at that width."""
    width = _width_of(pipeline, inputs)
    _record(
        pipeline,
        inputs,
        _metrics(len(inputs.flat_batch), width, fraction),
        _stats(step_time(width)),
    )
    return width


def test_low_acceptance_narrows_and_high_acceptance_widens() -> None:
    pipeline = _adaptive_pipeline()
    inputs = _decode_batches(4)
    assert _width_of(pipeline, inputs) == 7
    for _ in range(30):
        _step(pipeline, inputs, 0.0)
    assert _width_of(pipeline, inputs) == 1
    for _ in range(200):
        _step(pipeline, inputs, 1.0)
    assert _width_of(pipeline, inputs) == 7


def test_below_the_floor_a_narrow_width_rounds_up_to_a_captured_one() -> None:
    pipeline = _adaptive_pipeline(min_batch_size=10)
    for _ in range(30):
        _step(pipeline, _decode_batches(12), 0.0)
    # 9 and 10 share 12's bucket, but only 10 captured the narrow widths.
    assert _width_of(pipeline, _decode_batches(10)) == 1
    assert _width_of(pipeline, _decode_batches(9)) == 7


@pytest.mark.parametrize(
    ("steps_below_floor", "widens"), [(0, False), (40, True)]
)
def test_rounded_up_steps_below_the_floor_train_acceptance(
    steps_below_floor: int, widens: bool
) -> None:
    """Steps widened by the floor measure the drafts a narrow bucket skips."""
    pipeline = _adaptive_pipeline(min_batch_size=8)
    for _ in range(30):
        _step(pipeline, _decode_batches(16), 1 / 7)
    assert _width_of(pipeline, _decode_batches(16)) == 1
    for _ in range(steps_below_floor):
        assert _step(pipeline, _decode_batches(4), 1.0) == 7
    for _ in range(5):
        _step(pipeline, _decode_batches(16), 1.0)
    assert (_width_of(pipeline, _decode_batches(16)) > 1) is widens


def test_a_step_in_flight_across_a_switch_times_its_own_width() -> None:
    """Were a width-7 step's time filed under width 1, width 1 would look
    slow and the width would switch back."""
    pipeline = _adaptive_pipeline()
    inputs = _decode_batches(4)
    for _ in range(30):
        _step(pipeline, inputs, 0.0)
    assert _width_of(pipeline, inputs) == 1
    _record(pipeline, inputs, _metrics(4, 7, 0.0), _stats(100.0))
    for _ in range(9):
        _step(pipeline, inputs, 0.0)
    assert _width_of(pipeline, inputs) == 1


@pytest.mark.parametrize(
    ("early_sync_duration_s", "widens"), [(None, True), (0.01, False)]
)
def test_an_early_synced_step_is_not_timed(
    early_sync_duration_s: float | None, widens: bool
) -> None:
    pipeline = _adaptive_pipeline()
    inputs = _decode_batches(4)
    for _ in range(30):
        _step(pipeline, inputs, 0.0)
    assert _width_of(pipeline, inputs) == 1
    for _ in range(5):
        width = _width_of(pipeline, inputs)
        _record(
            pipeline,
            inputs,
            _metrics(4, width, 0.0),
            _stats(100.0, early_sync_duration_s),
        )
    assert (_width_of(pipeline, inputs) > 1) is widens


def test_a_mixed_batch_still_takes_the_mixed_width() -> None:
    pipeline = _adaptive_pipeline(mixed_width=2)
    assert _width_of(pipeline, _mixed_batch(4)) == 2


def test_every_width_is_captured_at_its_batch_size() -> None:
    """Graph capture probes only these widths; any other fails replay."""
    rng = random.Random(0)
    for _ in range(300):
        pipeline = _adaptive_pipeline(min_batch_size=rng.randrange(1, 33))
        for _ in range(rng.randrange(1, 60)):
            batch_size = rng.randrange(1, 33)
            width = _step(
                pipeline,
                _decode_batches(batch_size),
                rng.random(),
                lambda width: _steep(width) * rng.uniform(0.9, 1.1),
            )
            assert width in pipeline._widths_by_batch_size[batch_size]


@pytest.mark.parametrize(
    ("inputs", "metrics"),
    [
        (
            _decode_batches(4),
            _SpeculativeDecodingMetrics(
                num_speculative_tokens=7, num_verifications=0
            ),
        ),
        (_mixed_batch(3), _metrics(rows=4, width=7, fraction=0.0)),
    ],
    ids=["verified-nothing", "mixed-batch"],
)
def test_batches_without_decode_evidence_do_not_train(
    inputs: _Inputs, metrics: _SpeculativeDecodingMetrics
) -> None:
    pipeline = _adaptive_pipeline()
    for _ in range(50):
        _record(pipeline, inputs, metrics, _stats(_steep(7)))
    assert _width_of(pipeline, _decode_batches(4)) == 7


def _speculative_from_cli(argv: list[str]) -> SpeculativeConfig:
    """Drives the generated flags into a ``SpeculativeConfig``."""
    captured: dict[str, Any] = {}

    @click.command()
    @config_to_flag(SpeculativeConfig)
    def cli(**kwargs: Any) -> None:
        captured.update(kwargs)

    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    spec = PipelineArgs.from_flat_kwargs(**captured).speculative
    assert spec is not None
    return spec


@pytest.mark.parametrize("widths", ["1,3,5", "all"])
def test_the_generated_flag_reaches_the_speculative_config(
    widths: str,
) -> None:
    spec = _speculative_from_cli(
        [
            "--speculative-method",
            "eagle",
            "--num-speculative-tokens",
            "5",
            "--adaptive-speculative-widths",
            widths,
        ]
    )
    assert spec.adaptive_speculative_widths == widths


def test_the_flag_is_unset_by_default() -> None:
    spec = _speculative_from_cli(["--speculative-method", "eagle"])
    assert spec.adaptive_speculative_widths is None
    assert spec.adaptive_speculative_min_batch_size == 1


def test_the_generated_floor_flag_reaches_the_speculative_config() -> None:
    spec = _speculative_from_cli(
        [
            "--speculative-method",
            "eagle",
            "--num-speculative-tokens",
            "5",
            "--adaptive-speculative-widths",
            "all",
            "--adaptive-speculative-min-batch-size",
            "16",
        ]
    )
    assert spec.adaptive_speculative_min_batch_size == 16
