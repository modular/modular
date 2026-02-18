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
"""Unit test for building, compiling, and executing a Llama3 graph on GPU."""

from __future__ import annotations

from collections.abc import Callable

from max.driver import Device, load_devices, scan_available_devices
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.architectures.llama3_legacy.model import (
    Llama3Inputs,
    Llama3Model,
)
from max.pipelines.lib import ModelOutputs
from test_common.mocks import DummyPipelineConfig
from transformers.models.llama.configuration_llama import LlamaConfig


def test_build_compile_and_execute_llama3_graph(
    hf_config: LlamaConfig,
    weights: SafetensorWeights,
    make_pipeline_config: Callable[..., DummyPipelineConfig],
    make_kv_inputs: Callable[..., KVCacheInputs],
    make_inputs: Callable[..., Llama3Inputs],
    assert_model_outputs: Callable[
        [ModelOutputs, Llama3Inputs, list[Device]], None
    ],
) -> None:
    """Test building, compiling, and executing a Llama3 graph on GPU."""

    device_specs = scan_available_devices()[:1]
    pipeline_config = make_pipeline_config(device_specs)

    devices = load_devices(device_specs)
    session = InferenceSession(devices=devices)

    model = Llama3Model(
        pipeline_config=pipeline_config,
        session=session,
        huggingface_config=hf_config,
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights,
    )

    kv_inputs = make_kv_inputs(
        pipeline_config,
        session,
        device_refs=[DeviceRef.GPU()],
    )

    inputs = make_inputs(devices, kv_inputs)

    outputs = model.execute(inputs)
    assert_model_outputs(outputs, inputs, devices)
