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
"""Builds a tiny Laguna target on CPU, for no-GPU gates of its spec path.

The target is the parity suite's S-2.1-like checkpoint: full and sliding
attention, two query-head counts, a dense first layer and NVFP4 routed
experts. Nothing here executes: the stage only emits the ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from laguna_parity import CASES, MAX_LENGTH, _checkpoint, _hf_config, _weights
from max import functional as F
from max.driver import CPU, Buffer, DeviceSpec
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn import Module
from max.nn import _graph_ops as kernels
from max.pipelines.architectures.hf_config_shims import LagunaHFConfig
from max.pipelines.architectures.laguna_modulev3.laguna import Laguna
from max.pipelines.architectures.laguna_modulev3.model_config import (
    LagunaDecoderConfig,
)
from max.pipelines.architectures.laguna_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.module_registration import (
    _DecoderGraph,
    decoder_input_types,
)
from max.tensor import default_device, default_dtype
from test_common.mocks import DummyPipelineConfig

CASE = CASES["s21_like_nvfp4"]


def hf_config() -> LagunaHFConfig:
    """The tiny target's HuggingFace config."""
    return _hf_config(CASE)


def target_config(
    output: ModelOutputConfig | None = None,
) -> LagunaDecoderConfig:
    """The tiny target's config on CPU, finalized for ``output``."""
    hf = hf_config()
    pipeline_config = DummyPipelineConfig(
        model_path="poolside/Laguna-S-2.1-NVFP4",
        quantization_encoding="float4_e2m1fnx2",
        max_batch_size=2,
        max_length=MAX_LENGTH,
        device_specs=[DeviceSpec.cpu()],
        weight_path=[Path("fake.safetensors")],
        huggingface_config=hf,
    )
    config = LagunaDecoderConfig.initialize(
        pipeline_config, max_seq_len=MAX_LENGTH
    )
    checkpoint = _weights(
        _checkpoint(hf, CASE["bias_dtype"], CASE["nvfp4_layers"])
    )
    config.finalize(
        hf,
        convert_safetensor_state_dict(
            dict(checkpoint.items()), hf, pipeline_config
        ),
        output or ModelOutputConfig(),
    )
    return config


def fake_weights(module: Module[..., Any]) -> dict[str, WeightData]:
    """A zero-filled weight for every parameter, at its declared shape."""
    weights = {}
    for name, param in module.parameters:
        shape = [int(d) for d in param.shape]
        weights[name] = WeightData(
            name=name,
            dtype=param.dtype,
            shape=Shape(shape),
            data=Buffer.zeros(tuple(shape), dtype=param.dtype),
        )
    return weights


def build_decoder(config: LagunaDecoderConfig) -> Laguna:
    """The registered decoder, built as serving builds it."""
    with F.lazy(), default_dtype(config.dtype), default_device(CPU()):
        return Laguna(config)


def stage_decoder(decoder: Laguna) -> str:
    """The serving graph of ``decoder``, as its IR text."""
    graph = _DecoderGraph(decoder)
    with pytest.MonkeyPatch.context() as mp, F.lazy():
        # The stage only emits ops, so pin the SM100 paths the GPU build takes.
        mp.setattr(kernels, "_is_sm10x_gpu", lambda: True)
        staged, _ = graph.stage(
            decoder_input_types(decoder.kv_params, DeviceRef.CPU()),
            weights=fake_weights(graph),
        )
    return str(staged)
