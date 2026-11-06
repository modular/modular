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
"""Checks Gemma4 role weights, optional vision and late cache attachment."""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4_modulev3 import spec_adapters
from max.pipelines.architectures.gemma4_modulev3.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4_modulev3.model_config import (
    Gemma4Config,
)
from max.pipelines.architectures.gemma4_modulev3.modules import (
    Gemma4Decoder,
    Gemma4Vision,
)
from max.pipelines.architectures.gemma4_modulev3.registered_batch_processor import (
    _Gemma4Inputs,
)
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.speculative.driver import SequentialBatch


def test_raw_checkpoint_names_are_preserved_until_role_weights_are_prepared() -> (
    None
):
    value = NonCallableMock(spec=WeightData, dtype=DType.float32)
    raw = {
        "model.language_model.layers.0.input_layernorm.weight": value,
        "model.vision_tower.patch_embedder.weight": value,
        "model.embed_vision.embedding_projection.weight": value,
    }
    config = Gemma4Config.__new__(Gemma4Config)
    assert config.preserve_checkpoint_keys
    prepared = config.prepare_weights(raw)
    assert set(prepared) == {
        "decoder.language_model.layers.0.input_layernorm.weight",
        "vision.patch_embedder.weight",
        "vision.embed_vision.embedding_projection.weight",
    }
    assert all(weight is value for weight in prepared.values())
    assert "model.language_model.layers.0.input_layernorm.weight" in raw


def test_finalize_completes_a_copy_of_the_text_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The text config's rope_theta raises when read, as a full copy reads it.
    planning = Gemma4TextConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        hidden_activation="gelu_tanh",
        max_position_embeddings=8,
        max_seq_len=8,
        rms_norm_eps=1e-6,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=8,
        sliding_window=8,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=-1,
        dtype=DType.bfloat16,
        devices=[DeviceRef.CPU()],
        interleaved_rope_weights=False,
        kv_params=NonCallableMock(),
        layer_types=["sliding_attention"],
    )
    monkeypatch.setattr(
        Gemma4ForConditionalGenerationConfig,
        "finalize",
        lambda self, huggingface_config, state_dict, return_logits: None,
    )
    config = Gemma4Config.__new__(Gemma4Config)
    config.devices = [DeviceRef.CPU()]
    config.text_config = planning
    config.finalize(
        NonCallableMock(),
        {},
        ModelOutputConfig(
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        ),
    )
    assert config.text_config is not planning
    assert (
        config.text_config.return_hidden_states
        == ReturnHiddenStates.ALL_NORMALIZED
    )
    assert planning.return_hidden_states == ReturnHiddenStates.NONE


def test_unified_checkpoint_skips_only_the_vision_tower() -> None:
    config: Any = SimpleNamespace(vision_config=None)
    assert not Gemma4Vision.is_enabled(config)
    config.vision_config = object()
    assert Gemma4Vision.is_enabled(config)


def test_late_cache_attachments_pack_only_device_zero_vision() -> None:
    buffers = [Buffer.from_numpy(np.array([i], np.int64)) for i in range(7)]
    _, structure = tree.flatten(
        OrderedDict(
            tokens="tokens",
            vision=VisionInputs(
                embeddings="embeddings", scatter_indices="indices"
            ),
        )
    )
    inputs = _Gemma4Inputs(
        tokens=buffers[0],
        input_row_offsets=buffers[1],
        return_n_logits=buffers[2],
        input_tree_def=structure,
    )
    inputs.vision_embeddings = buffers[3:5]
    inputs.vision_scatter_indices = buffers[5:7]
    assert inputs.buffers == (buffers[0], buffers[3], buffers[5])


def test_the_spec_target_verifies_the_wrapped_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(spec_adapters, "tensor_compat_verify", lambda f: f)
    monkeypatch.setattr(
        spec_adapters,
        "_verify",
        lambda target, batch: calls.append((target, batch)),
    )
    language_model = object.__new__(Gemma4TextModel)
    decoder: Any = object.__new__(Gemma4Decoder)
    decoder.model = SimpleNamespace(language_model=language_model)
    verify = spec_adapters.verify(
        decoder, None, ReturnHiddenStates.ALL_NORMALIZED, None
    )
    batch = NonCallableMock(spec=SequentialBatch)
    verify(batch)
    assert calls == [(language_model, batch)]
