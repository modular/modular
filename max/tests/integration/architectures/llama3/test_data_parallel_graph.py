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
"""The data-parallel Llama 3 decoder graph, traced without a GPU."""

from __future__ import annotations

from collections.abc import Mapping

from max import tree
from max._artifact.value_types import treedef_to_json
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.graph_groups import Logits
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.llama3.data_parallel_llama import (
    DataParallelLlama,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config


def _config(devices: list[DeviceRef]) -> Llama3Config:
    return Llama3Config(
        hidden_size=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=1,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=64,
        intermediate_size=16,
        interleaved_rope_weights=False,
        vocab_size=32,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=4,
            head_dim=2,
            num_layers=1,
            devices=devices,
            data_parallel_degree=len(devices),
        ),
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        rms_norm_eps=1e-5,
        clip_qkv=None,
        norm_method="rms_norm",
        devices=devices,
    )


def test_data_parallel_decoder_joins_its_replicas_named_logits() -> None:
    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
    config = _config(devices)
    model = DataParallelLlama(config)
    model.state_dict()

    graph = model.trace(*model.input_types(config.kv_params))

    outputs = graph.output_tree
    assert isinstance(outputs, Mapping)
    assert list(outputs) == ["logits"]
    assert isinstance(outputs["logits"], Logits)
    (logits,) = graph.output_types
    assert isinstance(logits, TensorType)
    assert logits.device == devices[0]
    treedef_to_json(tree.flatten(outputs)[1])
