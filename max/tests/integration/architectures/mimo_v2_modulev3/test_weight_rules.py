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
"""The MiMo-V2 weight rules against the published checkpoint's tensor names.

One name per pattern from every safetensors file of
``XiaomiMiMo/MiMo-V2.6-Flash-RL``: the text model's index (decoder, MTP
layers, vision and audio towers), ``dflash/`` and ``audio_tokenizer/``, which
MAX's weight discovery globs in beside the decoder. The rules must keep the
decoder's tensors under the names it reads, relabel the routed experts'
``U8`` block scales as E8M0, and drop everything else.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.mimo_v2_modulev3.weight_rules import (
    mimo_v2_weight_rules,
)
from max.pipelines.lib.weight_transform import WeightTransform

# Checkpoint name -> the decoder's name.
_KEPT = {
    "lm_head.weight": "lm_head.weight",
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "model.layers.0.input_layernorm.weight": "layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight": (
        "layers.0.post_attention_layernorm.weight"
    ),
    "model.layers.0.self_attn.attention_sink_bias": (
        "layers.0.self_attn.attention_sink_bias"
    ),
    "model.layers.0.self_attn.o_proj.weight": "layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.qkv_proj.weight": (
        "layers.0.self_attn.qkv_proj.weight"
    ),
    "model.layers.0.self_attn.qkv_proj.weight_scale_inv": (
        "layers.0.self_attn.qkv_proj.weight_scale"
    ),
    "model.layers.0.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.gate_proj.weight_scale_inv": (
        "layers.0.mlp.gate_proj.weight_scale"
    ),
    "model.layers.0.mlp.up_proj.weight": "layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight_scale_inv": (
        "layers.0.mlp.down_proj.weight_scale"
    ),
    "model.layers.1.mlp.gate.weight": "layers.1.mlp.gate.gate_score.weight",
    "model.layers.1.mlp.gate.e_score_correction_bias": (
        "layers.1.mlp.gate.e_score_correction_bias"
    ),
    "model.layers.1.mlp.experts.7.gate_proj.weight": (
        "layers.1.mlp.experts.7.gate_proj.weight"
    ),
    "model.layers.1.mlp.experts.7.up_proj.weight_scale": (
        "layers.1.mlp.experts.7.up_proj.weight_scale"
    ),
    "model.layers.1.mlp.experts.7.down_proj.weight_scale": (
        "layers.1.mlp.experts.7.down_proj.weight_scale"
    ),
}

_DROPPED = (
    # The text model's index: MTP layers, vision and audio towers.
    "model.mtp.layers.0.eh_proj.weight",
    "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv",
    "visual.blocks.0.attn.qkv.weight",
    "visual.merger.ln_q.weight",
    "visual.patch_embed.proj.weight",
    "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight",
    "audio_encoder.projection.mlp.0.weight",
    "speech_embeddings.0.weight",
    # dflash/: the drafter, which loads as its own checkpoint.
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.mlp.down_proj.weight",
    # audio_tokenizer/.
    "encoder.layers.0.fc1.weight",
    "encoder.quantizer.vq.layers.0._codebook.embed",
    "decoder.layers.0.self_attn.out_proj.weight",
    "decoder.vocoder.istft.window",
)

_EXPERT_SCALES = {
    "layers.1.mlp.experts.7.up_proj.weight_scale",
    "layers.1.mlp.experts.7.down_proj.weight_scale",
}


def _weight(name: str) -> WeightData:
    # Routed-expert block scales are stored as U8; the rest are stand-ins.
    is_expert_scale = ".experts." in name and name.endswith(".weight_scale")
    dtype = np.uint8 if is_expert_scale else np.float32
    return WeightData.from_numpy(np.zeros((2,), dtype=dtype), name)


def test_the_rules_map_the_published_checkpoint() -> None:
    names = [*_KEPT, *_DROPPED]
    adapted = WeightTransform(mimo_v2_weight_rules()).apply_data(
        {name: _weight(name) for name in names}
    )

    assert set(adapted) == set(_KEPT.values())
    for name, data in adapted.items():
        want = DType.float8_e8m0fnu if name in _EXPERT_SCALES else DType.float32
        assert data.dtype == want, name
