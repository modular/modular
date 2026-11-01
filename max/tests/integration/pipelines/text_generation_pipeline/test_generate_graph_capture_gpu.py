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
"""``GenerateIterator`` on a real GPU pipeline, before and after capture warmup.

The same greedy batches run eagerly, then again after
``warmup_graph_capture()``, where every decode step replays a captured graph.
Both runs must produce the same tokens and logits within tolerance. Step
timings are logged, not asserted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import logging
import statistics
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures import register_all_models
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    KVCacheConfig,
    ModelOutputs,
    OverlapTextGenerationPipeline,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.generate import GenerateIterator
from max.pipelines.modeling.types import TextGenerationInputs

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT_LENGTHS = (5, 17, 40, 100)
MAX_NEW_TOKENS = 16
# The step after which the second scenario removes its last row, so later
# decode steps replay a smaller captured batch.
REMOVE_AFTER_STEP = 3
PASSAGE = (
    "The history of computing is a story of steadily shrinking machines and "
    "growing ambitions. Early mechanical calculators could add and subtract, "
    "but the first programmable electronic computers filled entire rooms and "
    "consumed enormous amounts of power. Transistors replaced vacuum tubes, "
    "integrated circuits replaced discrete transistors, and each generation "
    "of hardware made the previous one look slow and expensive. Personal "
    "computers brought this power into homes and offices during the late "
    "twentieth century, and the internet connected them into a global "
    "network. Today a phone in a pocket holds more memory than every "
    "computer of the nineteen sixties combined, and researchers continue to "
    "search for new ways to store, move, and process information faster "
    "while using less energy than before. "
)

Pipeline = OverlapTextGenerationPipeline[TextContext]


@dataclasses.dataclass
class ForwardSpy:
    """Counts forwards and copies their logits when asked."""

    calls: int = 0
    record_logits: bool = False
    logits: list[np.ndarray] = dataclasses.field(default_factory=list)

    def wrap(
        self, forward: Callable[..., ModelOutputs]
    ) -> Callable[..., ModelOutputs]:
        def spy(*args: object, **kwargs: object) -> ModelOutputs:
            self.calls += 1
            outputs = forward(*args, **kwargs)
            if self.record_logits:
                # Replay reuses its output buffer, so copy before the next
                # forward overwrites it.
                self.logits.append(
                    np.array(outputs.logits.to_numpy(), copy=True)
                )
            return outputs

        return spy


@dataclasses.dataclass
class ScenarioRun:
    sequences: list[list[int]]
    steps: int
    eager_forwards: int
    replayed_forwards: int
    decode_step_seconds: list[float]


@pytest.fixture(params=[True, False], ids=["synchronous", "overlap"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[Pipeline]:
    synchronous: bool = request.param
    register_all_models()
    config = PipelineConfig.from_args(
        PipelineArgs(
            model_path=MODEL,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding="bfloat16",
            max_length=512,
            kv_cache=KVCacheConfig(enable_prefix_caching=False),
            runtime=PipelineRuntimeConfig(
                max_batch_size=len(PROMPT_LENGTHS),
                device_graph_capture=True,
                disable_overlap=synchronous,
            ),
        )
    )
    _, retrieved = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(retrieved, OverlapTextGenerationPipeline)
    assert retrieved.overlap_active is not synchronous
    assert not retrieved.graph_capture_ready
    yield retrieved
    del retrieved
    gc.collect()


def _input_ids(pipeline: Pipeline) -> list[list[int]]:
    encoded = asyncio.run(
        pipeline.tokenizer.encode(PASSAGE, add_special_tokens=False)
    )
    tokens = [int(token) for token in encoded]
    assert len(tokens) >= max(PROMPT_LENGTHS)
    return [tokens[:length] for length in PROMPT_LENGTHS]


def _run_scenario(
    pipeline: Pipeline,
    input_ids: list[list[int]],
    eager: ForwardSpy,
    replay: ForwardSpy,
    remove_row: bool,
) -> ScenarioRun:
    eager_before, replay_before = eager.calls, replay.calls
    step_seconds: list[float] = []
    with GenerateIterator(
        pipeline,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        top_k=1,
        ignore_eos=True,
    ) as batch:
        while not batch.done:
            start = time.perf_counter()
            batch.step()
            step_seconds.append(time.perf_counter() - start)
            if remove_row and len(step_seconds) == REMOVE_AFTER_STEP:
                batch.remove(len(input_ids) - 1)
    return ScenarioRun(
        sequences=batch.sequences,
        steps=len(step_seconds),
        eager_forwards=eager.calls - eager_before,
        replayed_forwards=replay.calls - replay_before,
        decode_step_seconds=step_seconds[1:],
    )


def _run_pass(
    pipeline: Pipeline,
    input_ids: list[list[int]],
    eager: ForwardSpy,
    replay: ForwardSpy,
) -> list[ScenarioRun]:
    runs = [
        _run_scenario(pipeline, input_ids, eager, replay, remove_row)
        for remove_row in (False, True)
    ]
    if pipeline.overlap_active:
        # Sync the last forward so warmup and later passes start clean.
        pipeline.execute(TextGenerationInputs(batches=[[]]))
    return runs


def _graph_capture_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "max.pipelines"
        and record.levelno == logging.WARNING
        and "warmup_graph_capture()" in record.getMessage()
    ]


def _median_decode_seconds(runs: list[ScenarioRun]) -> float:
    return statistics.median(
        seconds for run in runs for seconds in run.decode_step_seconds
    )


def test_replay_matches_eager(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_ids = _input_ids(pipeline)
    eager = ForwardSpy(record_logits=True)
    replay = ForwardSpy(record_logits=True)
    model = pipeline._pipeline_model
    monkeypatch.setattr(model, "execute", eager.wrap(model.execute))

    eager_runs = _run_pass(pipeline, input_ids, eager, replay)
    eager_logits = eager.logits
    eager.record_logits = False
    eager_timing_runs = _run_pass(pipeline, input_ids, eager, replay)

    assert _graph_capture_warnings(caplog)
    for run in eager_runs:
        assert run.eager_forwards == run.steps
    caplog.clear()

    start = time.perf_counter()
    pipeline.warmup_graph_capture()
    warmup_seconds = time.perf_counter() - start
    assert pipeline.graph_capture_ready
    runner = pipeline._graph_capture_runner
    assert runner is not None
    monkeypatch.setattr(runner, "replay", replay.wrap(runner.replay))
    eager.logits = []
    eager.record_logits = True

    replay_runs = _run_pass(pipeline, input_ids, eager, replay)
    eager.record_logits = False
    replay.record_logits = False
    replay_timing_runs = _run_pass(pipeline, input_ids, eager, replay)

    assert _graph_capture_warnings(caplog) == []
    for eager_run, replay_run in zip(eager_runs, replay_runs, strict=True):
        assert replay_run.sequences == eager_run.sequences
        assert replay_run.steps == eager_run.steps
        # Only the prefill step runs eagerly.
        assert replay_run.eager_forwards == 1
        assert replay_run.replayed_forwards == replay_run.steps - 1

    # Prefill logits come from the eager spy, decode logits from the replay
    # spy, in forward order within each scenario.
    replay_logits: list[np.ndarray] = []
    eager_iter, replay_iter = iter(eager.logits), iter(replay.logits)
    for run in replay_runs:
        replay_logits.append(next(eager_iter))
        replay_logits.extend(
            next(replay_iter) for _ in range(run.replayed_forwards)
        )
    assert len(replay_logits) == len(eager_logits)
    max_abs_diff = 0.0
    for expected, actual in zip(eager_logits, replay_logits, strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-3)
        max_abs_diff = max(
            max_abs_diff, float(np.max(np.abs(actual - expected)))
        )

    mode = "overlap" if pipeline.overlap_active else "synchronous"
    with capsys.disabled():
        print(
            f"\n[{mode}] warmup_graph_capture: {warmup_seconds:.2f}s, "
            f"{len(eager_logits)} forwards compared, logits max abs diff "
            f"{max_abs_diff:.3e}, median decode step eager "
            f"{_median_decode_seconds(eager_timing_runs) * 1e3:.3f}ms vs "
            f"replay {_median_decode_seconds(replay_timing_runs) * 1e3:.3f}ms"
        )
