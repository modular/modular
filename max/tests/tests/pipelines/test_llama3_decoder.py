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

"""Tests that the registered Llama 3 decoder builds for each parallelism."""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MHAKVCacheParams
from max.nn.legacy.layer import Module
from max.pipelines.architectures.llama3 import decoder as decoder_module
from max.pipelines.architectures.llama3 import llama_arch
from max.pipelines.architectures.llama3.decoder import (
    Llama3Decoder,
    Llama3DecoderConfig,
    Llama3TargetDecoder,
)
from max.pipelines.architectures.qwen2 import qwen2_arch
from max.pipelines.lib import GraphRole
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _builds_module,
)
from max.pipelines.lib.registration_graph_api import (
    GraphApiDecoderPipelineModel,
    graph_name,
    supports_data_parallel,
)


def _config(
    devices: list[DeviceRef], data_parallel_degree: int = 1
) -> Llama3DecoderConfig:
    return Llama3DecoderConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=16,
        intermediate_size=16,
        interleaved_rope_weights=False,
        vocab_size=8,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=2,
            head_dim=4,
            num_layers=1,
            devices=tuple(devices),
            data_parallel_degree=data_parallel_degree,
        ),
        attention_multiplier=0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        rms_norm_eps=1e-5,
        data_parallel_degree=data_parallel_degree,
        huggingface_model_type="llama",
    )


def _checkpoint() -> dict[str, np.ndarray]:
    """Returns zero weights under the single-device decoder's names."""
    decoder = Llama3Decoder(_config([DeviceRef.CPU()]))
    return {
        name: np.zeros([int(d) for d in weight.shape], np.float32)
        for name, weight in decoder.raw_state_dict().items()
    }


def test_one_device_keeps_the_llama3_graph_name() -> None:
    decoder = Llama3Decoder(_config([DeviceRef.CPU()]))
    assert graph_name(decoder) == "llama3"
    assert "layers.0.self_attn.k_proj.weight" in decoder.raw_state_dict()


class _FakeDistributedLlama3(Module):
    """Stands in for the tensor-parallel model, which needs real GPUs."""

    def __init__(self, config: Llama3DecoderConfig) -> None:
        super().__init__()

    def __call__(self) -> None:
        raise NotImplementedError


def test_tensor_parallel_names_its_graph_by_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decoder_module, "_DistributedLlama3", _FakeDistributedLlama3
    )
    decoder = Llama3Decoder(_config([DeviceRef.GPU(0), DeviceRef.GPU(1)]))
    assert isinstance(decoder.model, _FakeDistributedLlama3)
    assert graph_name(decoder) == "llama"
    assert graph_name(Llama3Decoder) == "llama3"


def test_data_parallel_loads_the_checkpoint_into_every_replica() -> None:
    assert supports_data_parallel(Llama3Decoder)
    checkpoint = _checkpoint()
    decoder = Llama3Decoder(
        _config([DeviceRef.GPU(0), DeviceRef.GPU(1)], data_parallel_degree=2)
    )
    assert graph_name(decoder) == "llama3"

    decoder.load_state_dict(checkpoint)

    assert set(decoder.state_dict()) == {
        f"models.{replica}.{name}"
        for replica in range(2)
        for name in checkpoint
    }


def test_llama3_registers_the_decoder_that_builds_a_spec_target() -> None:
    assert dict(llama_arch.modules) == {GraphRole.DECODER: Llama3TargetDecoder}
    assert llama_arch.config is Llama3DecoderConfig
    model = llama_arch.pipeline_model
    assert isinstance(model, type)
    assert issubclass(model, GraphApiDecoderPipelineModel)
    assert graph_name(Llama3TargetDecoder) == "llama3"
    # Speculative decoding builds Llama 3's target through the hook.
    assert _builds_module(llama_arch)
    # Another architecture that registers the plain decoder gets no hook.
    assert not _builds_module(qwen2_arch)
