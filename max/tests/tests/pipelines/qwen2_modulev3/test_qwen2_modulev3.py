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
"""Tests the Qwen2 ModuleV3 registration, config and weight names."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import MHAKVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.qwen2_modulev3 import (
    Qwen2,
    Qwen2Config,
    qwen2_modulev3_arch,
)
from max.pipelines.architectures.qwen2_modulev3.model_config import (
    _rope_type,
)
from max.pipelines.architectures.qwen2_modulev3.weight_adapters import (
    max_weight_name,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh

_LAYERS = 2
_HF_NAMES = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    *(
        f"model.layers.{i}.{name}"
        for i in range(_LAYERS)
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    ),
]


def _config(mesh: DeviceMesh | None = None) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=32,
        interleaved_rope_weights=False,
        dtype=DType.float32,
        mesh=mesh or NonCallableMock(),
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=4,
            num_layers=_LAYERS,
            devices=[DeviceRef.CPU()],
        ),
    )


def _weight() -> WeightData:
    return WeightData.from_numpy(np.zeros((1,), np.float32), "w")


def test_registers_the_decoder_by_role() -> None:
    assert qwen2_modulev3_arch.name == "Qwen2ForCausalLM_ModuleV3"
    assert qwen2_modulev3_arch.modules == {GraphRole.DECODER: Qwen2}


def test_finalize_returns_a_finalized_copy() -> None:
    config = _config()
    finalized = config.finalize(
        SimpleNamespace(tie_word_embeddings=False),
        {"language_model.lm_head.weight": _weight()},
        ModelOutputConfig(return_logits=ReturnLogits.VARIABLE),
    )

    assert finalized is not config
    assert finalized.return_logits == ReturnLogits.VARIABLE
    assert not finalized.tie_word_embeddings
    assert config.return_logits == ReturnLogits.LAST_TOKEN


def test_finalize_ties_embeddings_without_an_lm_head() -> None:
    finalized = _config().finalize(
        SimpleNamespace(tie_word_embeddings=False), {}, ModelOutputConfig()
    )

    assert finalized.tie_word_embeddings


@pytest.mark.parametrize(
    "rope_scaling, expected",
    [
        (None, "default"),
        ({"rope_type": "default", "rope_theta": 1e6}, "default"),
        ({"type": "yarn", "factor": 4.0}, "yarn"),
    ],
)
def test_rope_type(
    rope_scaling: dict[str, object] | None, expected: str
) -> None:
    assert _rope_type(SimpleNamespace(rope_scaling=rope_scaling)) == expected


@pytest.mark.parametrize("tied", [True, False])
def test_adapted_checkpoint_names_match_the_decoder(tied: bool) -> None:
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    config = _config(mesh)
    hf_names = _HF_NAMES if tied else [*_HF_NAMES, "lm_head.weight"]
    state_dict = {max_weight_name(name): _weight() for name in hf_names}
    config = config.finalize(
        SimpleNamespace(tie_word_embeddings=tied),
        state_dict,
        ModelOutputConfig(),
    )

    with F.lazy(), default_dtype(DType.float32), default_device(mesh):
        decoder = Qwen2(config)

    assert {name for name, _ in decoder.parameters} == set(state_dict)
