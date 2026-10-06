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

"""Pins when the scheduler wraps ``pipeline.execute`` in a libkineto range."""

import queue
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.kv_cache import DummyKVCache
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from max.serve.config import KernelTraceLevel
from max.serve.scheduler import text_generation_scheduler
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.serve.telemetry import common


def _run_two_batches(
    monkeypatch: pytest.MonkeyPatch,
    level: KernelTraceLevel,
    events: list[tuple[object, ...]],
    execute_error: Exception | None = None,
    recording: bool = True,
) -> None:
    monkeypatch.setattr(common, "_kernel_trace_level", level)

    def range_begin_with_id(batch_id: int, name: str) -> int:
        events.append(("begin", batch_id, name))
        return batch_id + 100 if recording else 0

    monkeypatch.setattr(
        text_generation_scheduler, "range_begin_with_id", range_begin_with_id
    )
    monkeypatch.setattr(
        text_generation_scheduler,
        "range_end",
        lambda range_id: events.append(("end", range_id)),
    )

    def execute(inputs: TextGenerationInputs[TextContext]) -> dict[str, object]:
        events.append(("execute",))
        if execute_error is not None:
            raise execute_error
        return {}

    pipeline = Mock()
    pipeline.execute = Mock(side_effect=execute)
    pipeline._pipeline_model = Mock(_lora_manager=None)
    pipeline.extra_kv_managers = []
    request_queue: queue.Queue[TextContext | TextAndVisionContext] = (
        queue.Queue()
    )
    scheduler = TokenGenerationScheduler(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=4, target_tokens_per_batch_ce=32
        ),
        pipeline=pipeline,
        request_queue=request_queue,
        response_queue=queue.Queue(),
        cancel_queue=queue.Queue(),
        kv_cache=DummyKVCache(),
    )
    request_queue.put_nowait(
        TextContext(
            request_id=RequestID(),
            max_length=100,
            tokens=TokenBuffer(np.ones(8, dtype=np.int64)),
        )
    )
    scheduler._retrieve_pending_requests()
    for _ in range(2):
        scheduler._schedule(scheduler.batch_constructor.construct_batch())


def test_off_never_calls_the_range_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    _run_two_batches(monkeypatch, KernelTraceLevel.OFF, events)
    assert events == [("execute",), ("execute",)]


def test_batch_wraps_each_execute_with_its_batch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    _run_two_batches(monkeypatch, KernelTraceLevel.BATCH, events)
    assert events == [
        ("begin", 0, "max.batch"),
        ("execute",),
        ("end", 100),
        ("begin", 1, "max.batch"),
        ("execute",),
        ("end", 101),
    ]


def test_batch_skips_range_end_when_nothing_was_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    _run_two_batches(
        monkeypatch, KernelTraceLevel.BATCH, events, recording=False
    )
    assert events == [
        ("begin", 0, "max.batch"),
        ("execute",),
        ("begin", 1, "max.batch"),
        ("execute",),
    ]


def test_batch_closes_the_range_when_execute_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    with pytest.raises(RuntimeError):
        _run_two_batches(
            monkeypatch, KernelTraceLevel.BATCH, events, RuntimeError()
        )
    assert events == [("begin", 0, "max.batch"), ("execute",), ("end", 100)]
