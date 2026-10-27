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
"""Tests for the Gemma 4 per-layer embeddings (PLE).

Two properties the quantized (NVFP4) checkpoints depend on:

- the PLE projections are built with the model's quantization config, so the
  ``weight_scale`` / ``weight_scale_2`` / ``input_scale`` tensors a quantized
  checkpoint ships next to their weights have a declared owner (without it the
  strict state-dict load fails on unexpected keys); and
- the PLE placeholder mask covers the audio token, not only image and video,
  so an audio placeholder reads the pad row instead of a real embedding row.

These tests only build the module (no engine, no graph execution), so they run
on CPU.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4.layers.decoder_layer import (
    Gemma4TextDecoderLayer,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)

_HIDDEN = 64
_INTERMEDIATE = 128
_VOCAB = 256
_NUM_LAYERS = 2
_LAYER_TYPES = ["sliding_attention", "full_attention"]
_HEAD_DIM = 32
_GLOBAL_HEAD_DIM = 64
_N_HEADS = 2
_N_KV_HEADS = 2
_N_GLOBAL_KV_HEADS = 1
# Width of one layer's slice of the packed per-layer embedding table.
_PLE_DIM = 16

_IMAGE_TOKEN = 258_880
_VIDEO_TOKEN = 262_144
_AUDIO_TOKEN = 258_881
_PAD_TOKEN = 0

_RMS_NORM_EPS = 1e-6
_SLIDING_WINDOW = 8
_GLOBAL_ROPE_THETA = 1_000_000.0
_SLIDING_WINDOW_ROPE_THETA = 10_000.0
_GLOBAL_PARTIAL_ROTARY_FACTOR = 0.25
_FINAL_LOGIT_SOFTCAPPING = 30.0


def _make_config(
    *,
    quant_config: QuantConfig | None = None,
    audio_token_index: int | None = None,
) -> Gemma4ForConditionalGenerationConfig:
    """Build a small Gemma 4 config with per-layer embeddings enabled."""
    devices = [DeviceRef.CPU()]
    sliding_layers = _LAYER_TYPES.count("sliding_attention")
    global_layers = _LAYER_TYPES.count("full_attention")
    kv_params = MultiKVCacheParams.from_params(
        {
            "sliding_attention": MHAKVCacheParams(
                dtype=DType.bfloat16,
                n_kv_heads=_N_KV_HEADS,
                head_dim=_HEAD_DIM,
                num_layers=sliding_layers,
                devices=devices,
                page_size=128,
            ),
            "full_attention": MHAKVCacheParams(
                dtype=DType.bfloat16,
                n_kv_heads=_N_GLOBAL_KV_HEADS,
                head_dim=_GLOBAL_HEAD_DIM,
                num_layers=global_layers,
                devices=devices,
                page_size=128,
            ),
        }
    )
    text_config = Gemma4TextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=1024,
        max_seq_len=1024,
        rms_norm_eps=_RMS_NORM_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=_HEAD_DIM,
        sliding_window=_SLIDING_WINDOW,
        final_logit_softcapping=_FINAL_LOGIT_SOFTCAPPING,
        attn_logit_softcapping=None,
        rope_local_base_freq=_SLIDING_WINDOW_ROPE_THETA,
        sliding_window_pattern=-1,
        dtype=DType.bfloat16,
        devices=devices,
        interleaved_rope_weights=False,
        # The text config's own field holds the sliding-window cache params;
        # the per-layer-type tree lives on the top-level config.
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_layers=_NUM_LAYERS,
            devices=devices,
            page_size=128,
        ),
        num_global_key_value_heads=_N_GLOBAL_KV_HEADS,
        global_head_dim=_GLOBAL_HEAD_DIM,
        attention_k_eq_v=True,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=_GLOBAL_PARTIAL_ROTARY_FACTOR
        ),
        global_rope_theta=_GLOBAL_ROPE_THETA,
        sliding_window_rope_theta=_SLIDING_WINDOW_ROPE_THETA,
        layer_types=_LAYER_TYPES,
        hidden_size_per_layer_input=_PLE_DIM,
        vocab_size_per_layer_input=_VOCAB,
        pad_token_id=_PAD_TOKEN,
    )
    text_config.return_logits = ReturnLogits.LAST_TOKEN
    text_config.quant_config = quant_config

    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=DType.bfloat16,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=None,
        image_token_index=_IMAGE_TOKEN,
        video_token_index=_VIDEO_TOKEN,
        audio_token_index=audio_token_index,
        tie_word_embeddings=False,
    )


def _nvfp4_config() -> QuantConfig:
    """A modelopt NVFP4 config, as the quantized E-series checkpoints ship."""
    return QuantConfig.nvfp4(
        mlp_quantized_layers=set(range(_NUM_LAYERS)),
        attn_quantized_layers=set(),
    )


# ---------------------------------------------------------------------------
# PLE projections follow the model's quantization
# ---------------------------------------------------------------------------


def test_ple_projections_are_plain_when_the_model_is_not_quantized() -> None:
    model = Gemma4TextModel(_make_config())

    assert model.per_layer_model_projection.weight_scale is None
    assert model.per_layer_model_projection.weight_scale_2 is None
    assert model.per_layer_model_projection.input_scale is None
    assert model.per_layer_model_projection.weight.dtype == DType.bfloat16

    for layer in model.layers:
        assert layer.per_layer_input_gate.weight_scale is None
        assert layer.per_layer_projection.weight_scale is None
        assert layer.per_layer_input_gate.weight.dtype == DType.bfloat16

    keys = set(model.raw_state_dict())
    assert "per_layer_model_projection.weight" in keys
    assert "layers.0.per_layer_input_gate.weight" in keys
    assert not [key for key in keys if key.endswith(".weight_scale")]


def test_ple_projections_declare_quantized_weights_when_quantized() -> None:
    model = Gemma4TextModel(_make_config(quant_config=_nvfp4_config()))

    # The packed FP4 weight plus the scales the checkpoint carries next to
    # it: one per-16-input block scale, the per-tensor scale, and the static
    # activation scale.
    assert model.per_layer_model_projection.weight.dtype == DType.uint8
    assert model.per_layer_model_projection.weight_scale is not None
    assert model.per_layer_model_projection.weight_scale_2 is not None
    assert model.per_layer_model_projection.input_scale is not None

    for layer in model.layers:
        for projection in (
            layer.per_layer_input_gate,
            layer.per_layer_projection,
        ):
            assert projection.weight.dtype == DType.uint8
            assert projection.weight_scale is not None
            assert projection.weight_scale_2 is not None
            assert projection.input_scale is not None

    keys = set(model.raw_state_dict())
    for name in (
        "per_layer_model_projection.weight",
        "layers.0.per_layer_input_gate.weight",
        "layers.0.per_layer_projection.weight",
        "layers.1.per_layer_projection.weight",
    ):
        assert name in keys
        assert f"{name}_scale" in keys
        assert f"{name}_scale_2" in keys
        assert f"{name.rsplit('.', 1)[0]}.input_scale" in keys


def test_ple_projection_scale_shapes_match_the_packed_weight() -> None:
    """One FP4 block scale per 16 packed-away inputs."""
    model = Gemma4TextModel(_make_config(quant_config=_nvfp4_config()))

    model_projection = model.per_layer_model_projection
    assert model_projection.weight.shape == [
        _NUM_LAYERS * _PLE_DIM,
        _HIDDEN // 2,
    ]
    assert model_projection.weight_scale is not None
    assert model_projection.weight_scale.shape == [
        _NUM_LAYERS * _PLE_DIM,
        _HIDDEN // 16,
    ]

    # A ``LayerList`` indexes as its base ``Layer``, so narrow to the layer
    # this model actually stacks before reading its PLE projections.
    layer = model.layers[0]
    assert isinstance(layer, Gemma4TextDecoderLayer)

    per_layer = layer.per_layer_projection
    assert per_layer.weight.shape == [_HIDDEN, _PLE_DIM // 2]
    assert per_layer.weight_scale is not None
    assert per_layer.weight_scale.shape == [_HIDDEN, 1]

    input_gate = layer.per_layer_input_gate
    assert input_gate.weight.shape == [_PLE_DIM, _HIDDEN // 2]
    assert input_gate.weight_scale is not None
    assert input_gate.weight_scale.shape == [_PLE_DIM, _HIDDEN // 16]


# ---------------------------------------------------------------------------
# Placeholder mask
# ---------------------------------------------------------------------------


def test_placeholder_mask_covers_the_audio_token() -> None:
    model = Gemma4TextModel(_make_config(audio_token_index=_AUDIO_TOKEN))

    # ``_placeholder_token_ids`` is the set the per-layer embedding lookup
    # swaps for the pad row, so a real embedding row is never read for a
    # vision or audio placeholder.
    assert model._placeholder_token_ids == [
        _IMAGE_TOKEN,
        _VIDEO_TOKEN,
        _AUDIO_TOKEN,
    ]


def test_placeholder_mask_without_an_audio_token() -> None:
    model = Gemma4TextModel(_make_config(audio_token_index=None))

    assert model._placeholder_token_ids == [_IMAGE_TOKEN, _VIDEO_TOKEN]


@pytest.mark.parametrize("audio_token_index", [_AUDIO_TOKEN, None])
def test_placeholder_mask_includes_audio_only_when_declared(
    audio_token_index: int | None,
) -> None:
    """The mask is always the image and video tokens, plus audio when it
    exists; the pad row is what every one of them looks up."""
    model = Gemma4TextModel(_make_config(audio_token_index=audio_token_index))

    assert _IMAGE_TOKEN in model._placeholder_token_ids
    assert _VIDEO_TOKEN in model._placeholder_token_ids
    assert (_AUDIO_TOKEN in model._placeholder_token_ids) is (
        audio_token_index is not None
    )
    assert model._pad_token_id == _PAD_TOKEN
