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
"""Unit test for building a Llama3 graph on GPU."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

from max.driver import load_devices, scan_available_devices
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.llama3_legacy.model import (
    Llama3Model,
)
from test_common.mocks import DummyPipelineConfig
from transformers.models.llama.configuration_llama import LlamaConfig


def test_build_llama3_graph(
    hf_config: LlamaConfig,
    weights: SafetensorWeights,
    make_pipeline_config: Callable[..., DummyPipelineConfig],
) -> None:
    """Test building a Llama3 graph on GPU."""

    device_specs = scan_available_devices()[:1]
    pipeline_config = make_pipeline_config(device_specs)

    session = MagicMock(spec=InferenceSession)
    session.load.return_value = MagicMock(spec=Model, input_metadata=[])
    devices = load_devices(device_specs)

    Llama3Model(
        pipeline_config=pipeline_config,
        session=session,
        huggingface_config=hf_config,
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights,
    )

    session.load.assert_called()
