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
"""Tests the checks a config with ``runtime.eager`` set goes through."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from max.driver import DeviceSpec
from max.nn import Module
from max.pipelines.lib import MAXModelConfig, PipelineRuntimeConfig
from max.pipelines.lib.config.config import (
    _check_eager,
    _resolve_overlap_and_device_graph_capture,
)
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative.config import SpeculativeConfig


class _EagerDecoder(Module[..., object]):
    pass


class _GraphApiDecoder:
    pass


def _arch(decoder: type | None) -> Any:
    """Architecture stub that registers ``decoder`` as its decoder."""
    return SimpleNamespace(
        name="Stub",
        modules={} if decoder is None else {GraphRole.DECODER: decoder},
        pipeline_model=None,
        supports_overlap_scheduler=True,
        supports_device_graph_capture=True,
        supports_spec_decode_mixed_batches=False,
    )


@pytest.mark.parametrize(
    ("overrides", "conflict"),
    [
        ({"device_graph_capture": True}, "device_graph_capture"),
        (
            {"experimental_device_graph_synthesis": True},
            "experimental_device_graph_synthesis",
        ),
        ({"enable_overlap_scheduler": True}, "enable_overlap_scheduler"),
        ({"ep_size": 2}, "expert parallelism"),
        ({"export_mefs": "mefs"}, "export_mefs"),
    ],
)
def test_eager_rejects_runtime_settings_that_need_a_graph(
    overrides: dict[str, Any], conflict: str
) -> None:
    runtime = PipelineRuntimeConfig(eager=True, **overrides)
    with pytest.raises(ValueError, match=f"can't be combined with {conflict}"):
        _check_eager(runtime, None, _arch(_EagerDecoder), None, None)


def test_eager_rejects_lora_and_speculative_decoding() -> None:
    runtime = PipelineRuntimeConfig(eager=True)
    arch = _arch(_EagerDecoder)
    lora = LoRAConfig(enable_lora=True, lora_paths=["/some/lora/path"])
    speculative = SpeculativeConfig(speculative_method="eagle")
    with pytest.raises(ValueError, match="can't be combined with LoRA"):
        _check_eager(runtime, None, arch, lora, None)
    with pytest.raises(ValueError, match="speculative decoding"):
        _check_eager(runtime, None, arch, None, speculative)
    _check_eager(runtime, None, arch, LoRAConfig(enable_lora=False), None)


def test_eager_runs_text_generation_only() -> None:
    with pytest.raises(ValueError, match="embeddings_generation task"):
        _check_eager(
            PipelineRuntimeConfig(eager=True),
            None,
            _arch(_EagerDecoder),
            None,
            None,
            task=PipelineTask.EMBEDDINGS_GENERATION,
        )


@pytest.mark.parametrize(
    ("arch", "reason"),
    [
        (None, "needs a text generation model"),
        (_arch(_GraphApiDecoder), "is a graph API architecture"),
        (_arch(None), "builds its graphs in its own pipeline model"),
    ],
    ids=["none", "graph_api", "no_decoder"],
)
def test_eager_refuses_architectures_without_a_module_v3_decoder(
    arch: Any, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        _check_eager(PipelineRuntimeConfig(eager=True), None, arch, None, None)


def test_eager_runs_on_one_device() -> None:
    model = MAXModelConfig(
        model_path="test/model",
        device_specs=[DeviceSpec.accelerator(0), DeviceSpec.accelerator(1)],
    )
    with pytest.raises(ValueError, match="more than one device"):
        _check_eager(
            PipelineRuntimeConfig(eager=True),
            model,
            _arch(_EagerDecoder),
            None,
            None,
        )


def test_eager_turns_off_capture_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "max.pipelines.lib.config.config.accelerator_api",
        Mock(return_value="cuda"),
    )
    runtime = PipelineRuntimeConfig(eager=True, max_batch_size=16)
    model = MAXModelConfig(
        model_path="test/model", device_specs=[DeviceSpec.accelerator()]
    )
    capture, overlap = _resolve_overlap_and_device_graph_capture(
        runtime, SamplingConfig(), None, model, _arch(_EagerDecoder)
    )
    assert (capture, overlap) == (False, False)
