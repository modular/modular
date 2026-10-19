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
"""DeepSeek-V4.1-Flash loads its decoder graph spec around the weight loads."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock

import pytest
from max.pipelines.architectures.deepseekV41 import model as model_module
from max.pipelines.architectures.deepseekV41.model import (
    DROP_PAGE_CACHE_ENV,
    DeepseekV41Model,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.model import (
    UnifiedDSparkDeepseekV41Model,
)
from max.pipelines.lib import GraphRole


class _WeightLoads:
    """A routed-expert upload and an Engram pinning that records when each
    finishes; the pinning cannot finish before the compile does."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.compiled = threading.Event()

    def pin(self) -> None:
        assert self.compiled.wait(timeout=60)
        self.events.append("pin")

    def upload(self) -> None:
        self.events.append("upload")

    def compile(self) -> None:
        self.events.append("compiled")
        self.compiled.set()


def _stub_load(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_model: DeepseekV41Model,
    events: list[str],
) -> None:
    monkeypatch.setenv(DROP_PAGE_CACHE_ENV, "0")
    monkeypatch.setattr(pipeline_model, "_load_state_dict", dict)
    monkeypatch.setattr(
        pipeline_model, "_create_model_config", lambda state_dict: None
    )
    monkeypatch.setattr(
        pipeline_model,
        "_init_distributed_runtime",
        lambda session, model_config: None,
    )
    monkeypatch.setattr(
        pipeline_model,
        "_maybe_release_host_weights",
        lambda *models: events.append("released"),
    )
    monkeypatch.setattr(
        pipeline_model, "_wire_batch_processor", lambda model, config: None
    )
    pipeline_model.weights = Mock()


def test_load_graphs_overlaps_the_compile_with_the_engram_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The routed-expert upload finishes before the compile; the Engram
    pinning overlaps it and finishes before the host weights are released.
    """
    pipeline_model = object.__new__(DeepseekV41Model)
    events: list[str] = []
    loads = _WeightLoads(events)
    _stub_load(monkeypatch, pipeline_model, events)

    def trace() -> None:
        pipeline_model._defer_weight_load(loads.upload, overlaps_compile=False)
        pipeline_model._start_weight_loads()

    def graph_specs(model_config: object) -> dict[str, Any]:
        pipeline_model._defer_weight_load(loads.pin, overlaps_compile=True)
        return {GraphRole.DECODER: trace}

    decoder = Mock()

    def compile_and_init_graphs(
        session: object, specs: Mapping[str, Any]
    ) -> dict[str, Mock]:
        specs[GraphRole.DECODER]()
        loads.compile()
        return {GraphRole.DECODER: Mock(engine_model=decoder)}

    monkeypatch.setattr(
        model_module, "compile_and_init_graphs", compile_and_init_graphs
    )
    monkeypatch.setattr(pipeline_model, "graph_specs", graph_specs)

    assert pipeline_model.load_model(Mock()) is decoder
    assert events == ["upload", "compiled", "pin", "released"]


def test_dspark_still_loads_through_build_graph_for_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DSpark subclass keeps its own graph build, and its compile still
    overlaps the Engram pinning, which finishes before the weights init.
    """
    pipeline_model = object.__new__(UnifiedDSparkDeepseekV41Model)
    events: list[str] = []
    loads = _WeightLoads(events)
    _stub_load(monkeypatch, pipeline_model, events)
    graph = Mock()

    def build_graph_for_compile(
        session: object, state_dict: object, model_config: object
    ) -> tuple[Mock, dict[str, Any]]:
        pipeline_model._defer_weight_load(loads.pin, overlaps_compile=True)
        pipeline_model._defer_weight_load(loads.upload, overlaps_compile=False)
        return graph, {}

    def compile_reusing_mefs(compiled_graph: object) -> Mock:
        assert compiled_graph is graph
        loads.compile()
        return Mock()

    decoder = Mock()

    def init(compiled: object, weights_registry: object) -> Mock:
        events.append("init")
        return decoder

    session = Mock(compile_reusing_mefs=compile_reusing_mefs, init=init)
    monkeypatch.setattr(
        pipeline_model, "_build_graph_for_compile", build_graph_for_compile
    )

    assert pipeline_model.load_model(session) is decoder
    assert events == ["upload", "compiled", "pin", "init", "released"]
