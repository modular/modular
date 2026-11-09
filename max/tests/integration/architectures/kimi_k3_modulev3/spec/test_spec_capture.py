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
"""The captured spec pipeline's per-step replay check.

The spec pipeline is parity-tested EAGER; this is the one check its
captured form must satisfy. The MLA prefill plans take their extent from
the host pair the K3 spec adapter threads as a trailing input
(``dspark/mla_extent.py``) instead of reading the device plan back -- a
D2H sync no captured graph can hold. The pair selects in-graph between
the batch's exact extent (eager prompt/mixed steps, never captured) and
a capture-safe bound keyed on host inputs that follow the capture key
(captured decode steps), so the check grades the bound arm replaying
bit-for-bit what eager execution computes.

``test_replay_matches_eager_bit_exact_per_step`` -- replay == eager per
step, bit-exact, at the same ALIGNED batch characteristics: the spec
outputs, any draft distributions, the KV rows each request has written
(gathered through each arm's own lookup table, so a replay that writes a
page wrong or skips a write fails even when the outputs happen to agree)
and the recurrent-state pool rows each step's row ids name. Comparing
against unaligned eager is not sound: the MLA split-K count follows the
capture bucket. Raw pool and page-table buffers are deliberately not
compared -- the block pool is FIFO, so a second run over one pipeline
lands in different pages; the logical state, the rows each run itself
wrote, is the invariant.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import spec_parity
from _tiny_pair import (
    DRAFT_WIDTH,
    MAX_LENGTH,
    TinyPair,
    build_tiny_pair,
    random_prompt_rows,
)
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputsPerDevice,
    RecurrentStateInputsPerDevice,
)
from max.nn.kv_cache.input_types import RecurrentLeafInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    OverlapTextGenerationPipeline,
    UnifiedEagleOutputs,
)
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner

_PROMPT_LENGTHS = (3, 60, 124)
"""Prompt rows chosen so a run's decode lengths cross both the 64-token MLA
probe bucket and the 128-token page boundary: the 60-token row crosses 64
within its first decode steps, the 124-token row crosses 128."""

_PARITY_STEPS = 16
"""Steps the per-step bit-exact comparison drives: enough to pass a bucket
boundary without paying a full run's snapshot traffic."""

_Pipeline = OverlapTextGenerationPipeline[TextContext]
"""The concrete pipeline ``spec_parity._pipeline`` builds for this pair."""


# ===--------------------------------------------------------------------=== #
# Fixtures and pipeline builders
# ===--------------------------------------------------------------------=== #


class _CaptureUnavailable(Exception):
    """Capture warmup could not record the spec graph at all."""


@pytest.fixture(scope="module")
def tiny_pair(tmp_path_factory: pytest.TempPathFactory) -> TinyPair:
    return build_tiny_pair(
        tmp_path_factory.mktemp("k3_capture_pair"),
        random_prompt_rows(_PROMPT_LENGTHS),
    )


@dataclass
class CapturedPipeline:
    """A spec pipeline built with capture on, and its warmup verdict.

    ``warmup_error`` holds the exception text when warmup failed, so the
    test can fail on it in its own call phase rather than in fixture setup,
    where it would report as an error instead.
    """

    pipeline: _Pipeline
    runner: ServeGraphCaptureRunner | None
    warmup_error: str | None


def _warmup(pipeline: _Pipeline) -> str | None:
    """Runs graph-capture warmup, returning its error text if it raised."""
    try:
        pipeline.warmup_graph_capture()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _spec_pipeline(tiny_pair: TinyPair) -> _Pipeline:
    """Builds the tiny pair's spec pipeline with capture on.

    An explicit ``device_graph_capture=True`` captures whatever the arch
    flag says, so the check holds the fused pipeline to the captured form
    on this build alone.
    """
    pipeline = spec_parity._pipeline(
        tiny_pair.target,
        tiny_pair.drafter,
        1,
        MAX_LENGTH,
        DRAFT_WIDTH,
        None,
        True,
        prefer_module_v3=True,
        trust_remote_code=True,
        prompts=list(tiny_pair.prompt_rows),
        speculative=True,
    )
    assert isinstance(pipeline, OverlapTextGenerationPipeline)
    return pipeline


def _captured_verdict(pipeline: _Pipeline) -> CapturedPipeline:
    """Wraps a built pipeline with its warmup verdict."""
    warmup_error = _warmup(pipeline)
    # The runner only exists once warmup installs it, so read it after.
    return CapturedPipeline(
        pipeline, pipeline._graph_capture_runner, warmup_error
    )


@pytest.fixture(scope="module")
def captured_pair(tiny_pair: TinyPair) -> CapturedPipeline:
    """The pair served with capture forced on, warmup verdict included."""
    return _captured_verdict(_spec_pipeline(tiny_pair))


def _require_capture(captured: CapturedPipeline) -> None:
    """Fails the test with the warmup failure instead of a fixture error."""
    if captured.warmup_error is not None:
        raise _CaptureUnavailable(
            "device graph capture warmup failed: " + captured.warmup_error
        )
    assert captured.runner is not None


# ===--------------------------------------------------------------------=== #
# Logical state: leaves, pages and pools
# ===--------------------------------------------------------------------=== #


def _cache_leaves(
    value: object, path: str = ""
) -> Iterator[tuple[str, object]]:
    """Walks a step's cache-input tree, yielding every leaf with its path.

    The tree nests group names and per-device lists, so the walk is
    structural rather than keyed: whatever the nesting, every
    ``KVCacheInputsPerDevice`` and ``RecurrentStateInputsPerDevice`` comes
    out, and both arms of a comparison walk the same declaration order.
    """
    if isinstance(
        value, (KVCacheInputsPerDevice, RecurrentStateInputsPerDevice)
    ):
        yield path, value
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _cache_leaves(child, f"{path}[{index}]")
    elif tree.is_node(value):
        children, keys, _ = tree.flatten_one_level(value)
        for child, key in zip(
            children, keys or ("",) * len(children), strict=True
        ):
            yield from _cache_leaves(child, f"{path}/{key}" if key else path)


_NUMPY_VIEW_DTYPES = {
    DType.bfloat16: DType.uint16,
    DType.float8_e4m3fn: DType.uint8,
    DType.float8_e5m2: DType.uint8,
}
"""Numpy has no bfloat16 or fp8, so their buffers are read as raw bits --
which is all a bit-exact comparison wants of them anyway."""


def _to_numpy_bits(buffer: Buffer) -> np.ndarray:
    """A buffer's contents as numpy, bit-viewed where numpy lacks the dtype."""
    view_dtype = _NUMPY_VIEW_DTYPES.get(buffer.dtype)
    if view_dtype is not None:
        return buffer.view(dtype=view_dtype).to_numpy()
    return buffer.to_numpy()


def _written_rows(
    leaf: KVCacheInputsPerDevice[Buffer, Buffer], accepted: np.ndarray
) -> np.ndarray:
    """Gathers the KV rows this step leaves written, via the leaf's LUT.

    Every request's rows up to its post-step length -- prompt rows the CE
    step wrote plus whatever this step committed -- read out of the leaf's
    own pages, so two arms are compared on what they wrote, never on where
    the FIFO block pool happened to place them. Unwritten slack the plan
    zero-fills and the attention never reads is left out.
    """
    pool = _to_numpy_bits(leaf.kv_blocks)
    lut = leaf.lookup_table.to_numpy()
    lengths = leaf.cache_lengths.to_numpy().astype(np.int64)
    slots = pool.shape[3]
    written = np.minimum(
        lengths + np.atleast_1d(accepted).astype(np.int64).reshape(-1) + 1,
        lut.shape[1] * slots,
    )
    gathered = []
    for row, length in enumerate(written):
        tokens = np.arange(int(length))
        pages = lut[row, tokens // slots]
        # The pool blocks are [kv_dim, num_layers, slots, heads, head_dim]
        # (``KVCacheParams.shape_per_block``), so the slot position is the
        # pool's third axis, after the layer axis.
        gathered.append(pool[pages, :, :, tokens % slots, :, :])
    return np.concatenate(gathered)


def _pool_rows(leaf: RecurrentLeafInputs[Buffer, Buffer]) -> np.ndarray:
    """Gathers the recurrent-state rows this batch's row ids name.

    The live KDA pool and the verify ring both address their rows through
    ``live_row_ids``, so gathering by those ids compares each arm's own
    slots -- the state it wrote -- rather than raw pool bytes, which hold
    whatever earlier requests left behind them.
    """
    pool = _to_numpy_bits(leaf.pool)
    ids = leaf.live_row_ids.to_numpy()
    return pool[ids]


# ===--------------------------------------------------------------------=== #
# Replay spies
# ===--------------------------------------------------------------------=== #


@dataclass
class _StepRecord:
    """One step's outputs and written state, copied to host."""

    eager: bool
    characteristics: BatchCharacteristics
    outputs: list[np.ndarray]
    pages: list[np.ndarray]
    pools: list[np.ndarray]


class _ReplayRecorder:
    """Counts replays, and copies each step's outputs and state to host.

    Installed over ``runner.replay`` once per test, then driven run by run:
    with :attr:`eager` false it wraps the real replay; with it true it
    replaces replay with an eager execution of the very ``model_inputs``
    the replay path prepared. The pipeline builds those for the ALIGNED
    characteristics, so the eager run is eager-at-aligned -- the bit-exact
    comparison the per-step check needs, where comparing against unaligned
    eager would price the split-K difference instead of the capture. Each
    record stamps the mode it ran under.
    """

    def __init__(
        self,
        runner: ServeGraphCaptureRunner,
        pipeline: _Pipeline,
    ) -> None:
        # The bound replay, taken before the recorder replaces the runner's
        # attribute -- calling the attribute back would recurse through
        # this very ``__call__``.
        self._replay = runner.replay
        self._pipeline = pipeline
        self.eager = False
        self.calls = 0
        self.records: list[_StepRecord] = []

    def __call__(
        self,
        *,
        model_inputs: ModelInputs,
        batch_characteristics: BatchCharacteristics,
        debug_verify_replay: bool = False,
        debug_verify_model_inputs: ModelInputs | None = None,
    ) -> ModelOutputs:
        self.calls += 1
        # The launch-trace verification is not driven here: it executes the
        # step eagerly over the live state pools, which advances the KDA
        # state the replay would then advance again.
        assert not debug_verify_replay, (
            f"replay {self.calls} asked for the launch-trace verification,"
            " which this comparison does not drive"
        )
        if self.eager:
            outputs = self._pipeline._pipeline_model.execute(
                model_inputs=model_inputs
            )
        else:
            outputs = self._replay(
                model_inputs=model_inputs,
                batch_characteristics=batch_characteristics,
            )
        assert isinstance(outputs, UnifiedEagleOutputs)
        accepted = outputs.num_accepted_draft_tokens.to_numpy()
        records: list[np.ndarray] = [
            accepted.copy(),
            outputs.next_tokens.to_numpy().copy(),
            outputs.next_draft_tokens.to_numpy().copy(),
        ]
        # The graph returns the drafter's distributions only under
        # ``draft_proposal="sampled"``; greedy K3 returns none, so this
        # list stays empty there and the bit-exact set is the three
        # outputs. The verify's own logits are consumed in-graph and
        # returned by no surface; the next step's KV rows stand in for
        # them.
        if outputs.next_draft_probs_full is not None:
            records.append(outputs.next_draft_probs_full.to_numpy().copy())
        for replica in outputs.next_draft_probs_replicas or ():
            records.append(replica.to_numpy().copy())

        pages: list[np.ndarray] = []
        pools: list[np.ndarray] = []
        for _, leaf in _cache_leaves(model_inputs.kv_cache_inputs):
            if isinstance(leaf, KVCacheInputsPerDevice):
                pages.append(_written_rows(leaf, accepted))
            elif isinstance(leaf, RecurrentStateInputsPerDevice):
                for state_leaf in leaf.leaves:
                    pools.append(_pool_rows(state_leaf))
        self.records.append(
            _StepRecord(
                eager=self.eager,
                characteristics=batch_characteristics,
                outputs=records,
                pages=pages,
                pools=pools,
            )
        )
        return outputs


def _install_recorder(
    captured: CapturedPipeline, monkeypatch: pytest.MonkeyPatch
) -> _ReplayRecorder:
    """Counts and records every replay the pipeline makes from here on."""
    assert captured.runner is not None
    recorder = _ReplayRecorder(captured.runner, captured.pipeline)
    monkeypatch.setattr(captured.runner, "replay", recorder)
    return recorder


def _greedy_tokens(
    pipeline: _Pipeline,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
) -> list[list[int]]:
    """One greedy run's committed tokens per prompt."""
    tokens, _, _, _ = spec_parity._generate(
        pipeline, [list(row) for row in prompts], max_new_tokens, logits=False
    )
    return tokens


def _recorded_run(
    recorder: _ReplayRecorder,
    pipeline: _Pipeline,
    prompts: Sequence[Sequence[int]],
    *,
    eager: bool,
    max_new_tokens: int,
) -> list[list[int]]:
    """Drives one greedy run, its steps stamped with the run's mode."""
    recorder.eager = eager
    return _greedy_tokens(pipeline, prompts, max_new_tokens)


def _compare_step_records(recorder: _ReplayRecorder) -> None:
    """Bit-exact per-step comparison of a recorder's replay and eager runs."""
    replayed = [record for record in recorder.records if not record.eager]
    eager = [record for record in recorder.records if record.eager]
    assert replayed, "no step replayed a captured graph"
    assert len(replayed) == len(eager), (
        f"replay executed {len(replayed)} captured steps where the eager"
        f" run executed {len(eager)}"
    )
    for step, (got, want) in enumerate(zip(replayed, eager, strict=True)):
        context = f"captured step {step}"
        assert got.characteristics == want.characteristics, context
        assert len(got.outputs) == len(want.outputs), context
        for index, (got_out, want_out) in enumerate(
            zip(got.outputs, want.outputs, strict=True)
        ):
            np.testing.assert_array_equal(
                got_out,
                want_out,
                err_msg=(
                    f"{context}: spec output {index} (num_accepted, next"
                    " tokens, next drafts, any draft distributions) differs"
                    " between replay and eager"
                ),
            )
        assert len(got.pages) == len(want.pages), context
        for index, (got_pages, want_pages) in enumerate(
            zip(got.pages, want.pages, strict=True)
        ):
            np.testing.assert_array_equal(
                got_pages,
                want_pages,
                err_msg=(
                    f"{context}: the KV rows written so far (leaf {index},"
                    " gathered through each arm's own lookup table) differ"
                    " between replay and eager"
                ),
            )
        assert len(got.pools) == len(want.pools), context
        for index, (got_pool, want_pool) in enumerate(
            zip(got.pools, want.pools, strict=True)
        ):
            np.testing.assert_array_equal(
                got_pool,
                want_pool,
                err_msg=(
                    f"{context}: recurrent-state rows {index} (a KDA live"
                    " or ring pool, gathered by this batch's row ids)"
                    " differ between replay and eager"
                ),
            )


# ===--------------------------------------------------------------------=== #
# Replay == eager, bit-exact per step, at aligned characteristics
# ===--------------------------------------------------------------------=== #


@pytest.mark.compile_on_gpu(
    "capture warmup needs a real KV manager; virtual devices get a mock"
)
def test_replay_matches_eager_bit_exact_per_step(
    tiny_pair: TinyPair,
    captured_pair: CapturedPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay and eager-at-aligned must agree bit for bit on every step.

    Two greedy runs over the same pipeline: one replaying the captured
    graphs, one executing the very ``model_inputs`` the replay path
    prepared -- aligned dispatch metadata, same pool buffers -- eagerly.
    Each run writes its own pages (the block pool is FIFO, so the second
    run lands elsewhere), so the comparison gathers what each arm wrote
    -- every request's KV rows through that arm's own lookup table, and
    the recurrent-state rows its own row ids name -- which a skipped or
    doubled state write breaks even when the step outputs agree.

    The verify logits are the one piece of the step no surface returns:
    the driver consumes them in-graph (acceptance, the Markov chain), and
    the graph returns only the three unified buffers; the next step's
    KV rows, which those logits decided, are compared bit-exact in
    their place -- the accepted proxy for them, no product hook.
    """
    _require_capture(captured_pair)
    recorder = _install_recorder(captured_pair, monkeypatch)
    _recorded_run(
        recorder,
        captured_pair.pipeline,
        tiny_pair.prompt_rows,
        eager=False,
        max_new_tokens=_PARITY_STEPS,
    )
    _recorded_run(
        recorder,
        captured_pair.pipeline,
        tiny_pair.prompt_rows,
        eager=True,
        max_new_tokens=_PARITY_STEPS,
    )
    _compare_step_records(recorder)
