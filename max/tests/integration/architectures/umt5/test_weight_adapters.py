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

from __future__ import annotations

import numpy as np
from max.graph.weights import WeightData
from max.pipelines.architectures.umt5.model_config import UMT5ConfigBase
from max.pipelines.architectures.umt5.weight_adapters import (
    convert_safetensor_state_dict,
)


def _embedding_weight(name: str) -> WeightData:
    return WeightData.from_numpy(np.ones((4, 8), dtype=np.float32), name)


def test_convert_safetensor_state_dict_adds_shared_weight() -> None:
    encoder_weight = _embedding_weight("encoder.embed_tokens.weight")

    new_state_dict = convert_safetensor_state_dict(
        {"encoder.embed_tokens.weight": encoder_weight}
    )

    assert "encoder.embed_tokens.weight" in new_state_dict
    assert "shared.weight" in new_state_dict
    assert new_state_dict["shared.weight"].name == "shared.weight"
    assert new_state_dict["shared.weight"].dtype == encoder_weight.dtype
    assert new_state_dict["shared.weight"].shape == encoder_weight.shape


def test_convert_safetensor_state_dict_adds_encoder_weight() -> None:
    shared_weight = _embedding_weight("shared.weight")

    new_state_dict = convert_safetensor_state_dict(
        {"shared.weight": shared_weight}
    )

    assert "shared.weight" in new_state_dict
    assert "encoder.embed_tokens.weight" in new_state_dict
    assert (
        new_state_dict["encoder.embed_tokens.weight"].name
        == "encoder.embed_tokens.weight"
    )
    assert (
        new_state_dict["encoder.embed_tokens.weight"].dtype
        == shared_weight.dtype
    )
    assert (
        new_state_dict["encoder.embed_tokens.weight"].shape
        == shared_weight.shape
    )


def test_convert_safetensor_state_dict_raises_without_embeddings() -> None:
    non_embedding_weight = _embedding_weight(
        "encoder.block.0.layer.0.SelfAttention.q.weight"
    )
    try:
        convert_safetensor_state_dict(
            {
                "encoder.block.0.layer.0.SelfAttention.q.weight": (
                    non_embedding_weight
                )
            }
        )
    except ValueError as exc:
        assert "Missing UMT5 embedding weights" in str(exc)
        assert "shared.weight" in str(exc)
        assert "encoder.embed_tokens.weight" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing UMT5 embeddings.")


def test_umt5_config_defaults_match_wan_text_encoder() -> None:
    # Wan2.2-T2V-A14B-Diffusers text_encoder/config.json defaults.
    config = UMT5ConfigBase()
    assert config.classifier_dropout == 0.0
    assert config.vocab_size == 256384
    assert config.d_model == 4096
    assert config.d_kv == 64
    assert config.d_ff == 10240
    assert config.num_layers == 24
    assert config.num_decoder_layers == 24
    assert config.num_heads == 64
    assert config.relative_attention_num_buckets == 32
    assert config.relative_attention_max_distance == 128
    assert config.dropout_rate == 0.1
    assert config.layer_norm_epsilon == 1e-6
    assert config.initializer_factor == 1.0
    assert config.feed_forward_proj == "gated-gelu"
    assert config.use_cache is True
    assert config.output_past is True
    assert config.is_encoder_decoder is True
    assert config.decoder_start_token_id == 0
    assert config.pad_token_id == 0
    assert config.eos_token_id == 1
    assert config.tokenizer_class == "T5Tokenizer"
    assert config.tie_word_embeddings is False
    assert config.scalable_attention is True
