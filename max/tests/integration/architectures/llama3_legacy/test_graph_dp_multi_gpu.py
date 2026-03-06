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
"""Unit test for building, compiling, and executing a data-parallel Llama3 graph on GPU."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from max.driver import Buffer, Device, load_devices, scan_available_devices
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.llama3.model import (
    Llama3Inputs,
    Llama3Model,
)
from max.pipelines.lib import ModelOutputs
from test_common.mocks import DummyPipelineConfig
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig


def weights_from_hf_config_dp(hf_config: LlamaConfig) -> SafetensorWeights:
    """Auto-generate weights for the data-parallel code path.

    The DP code path (create_data_parallel_graph) prepends "model." to all
    state dict keys before loading.  HuggingFace already includes a "model."
    prefix, so we strip it here to avoid double-prefixing.

    We also pass ``tensors`` explicitly so that ``SafetensorWeights.items()``
    can iterate the weight map (with empty filepaths the internal ``_tensors``
    set would otherwise be empty).  Values are stored as MAX Buffers so that
    dtype comparisons in ``load_state_dict(strict=True)`` succeed.
    """
    hf_model = LlamaForCausalLM(hf_config)
    weight_map = {}
    for name, param in hf_model.named_parameters():
        key = name.removeprefix("model.")
        weight_map[key] = Buffer.from_dlpack(
            param.detach().zero_().to(torch.bfloat16)
        )
    return SafetensorWeights(
        [],
        tensors=set(weight_map.keys()),
        tensors_to_file_idx={k: 0 for k in weight_map},
        _st_weight_map=weight_map,
    )


def test_build_compile_and_execute_dp_llama3_graph(
    hf_config: LlamaConfig,
    make_pipeline_config: Callable[..., DummyPipelineConfig],
    make_kv_inputs: Callable[..., KVCacheInputs],
    make_inputs: Callable[..., Llama3Inputs],
    assert_model_outputs: Callable[
        [ModelOutputs, Llama3Inputs, list[Device]], None
    ],
) -> None:
    """Test building, compiling, and executing a data-parallel Llama3 graph on GPU."""

    device_specs = scan_available_devices()[:2]
    if len(device_specs) < 2:
        pytest.skip("Test requires 2 GPUs")

    pipeline_config = make_pipeline_config(device_specs, max_batch_size=2)
    pipeline_config.model.data_parallel_degree = 2

    devices = load_devices(device_specs)
    session = InferenceSession(devices=devices)

    model = Llama3Model(
        pipeline_config=pipeline_config,
        session=session,
        devices=devices,
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights_from_hf_config_dp(hf_config),
    )

    kv_inputs = make_kv_inputs(
        pipeline_config,
        session,
        device_refs=[DeviceRef.from_device(device) for device in devices],
        data_parallel_degree=2,
        num_replicas=2,
        total_num_pages=2,
    )

    # data_parallel_splits: cumulative batch counts per replica.
    # [0, 1, 2] means replica 0 has 1 batch item, replica 1 has 1.
    data_parallel_splits = Buffer.from_numpy(
        np.array([0, 1, 2], dtype=np.int64)
    )

    inputs = make_inputs(
        devices,
        kv_inputs,
        num_batches=2,
        data_parallel_splits=data_parallel_splits,
    )

    outputs = model.execute(inputs)
    assert_model_outputs(outputs, inputs, devices)
