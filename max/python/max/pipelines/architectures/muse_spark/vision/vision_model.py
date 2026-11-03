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

"""Muse Spark vision tower, adapter and projection (ModuleV3 API)."""

from __future__ import annotations

from max.pipelines.architectures.muse_glimmer.model_config import (
    MuseGlimmerVisionConfig,
)
from max.pipelines.architectures.muse_glimmer.vision.vision_model import (
    MuseGlimmerVisionModel,
)
from transformers import AutoConfig


class MuseSparkVisionModel(MuseGlimmerVisionModel):
    """Maps packed image patches to normed text-width image embeddings.

    Muse Glimmer's modules; Spark differs only in its host-side inputs.
    """


def parse_vision_config(
    huggingface_config: AutoConfig,
) -> MuseGlimmerVisionConfig:
    """Builds the vision config from the top-level HuggingFace config.

    Raises:
        ValueError: If the vision RoPE is not ``axial``, the only kind the
            reference tower accepts, or the merged width disagrees with
            ``out_hidden_size``.
    """
    hf_vision = huggingface_config.vision_config
    rope = hf_vision.rope_parameters
    if rope.get("rope_type") != "axial":
        raise ValueError(f"Vision rope parameters {rope} not supported")
    config = MuseGlimmerVisionConfig(
        hidden_size=hf_vision.hidden_size,
        intermediate_size=hf_vision.intermediate_size,
        num_hidden_layers=hf_vision.num_hidden_layers,
        num_attention_heads=hf_vision.num_attention_heads,
        hidden_act=hf_vision.hidden_act,
        layer_norm_eps=hf_vision.layer_norm_eps,
        layer_types=list(hf_vision.layer_types),
        patch_size=hf_vision.patch_size,
        patch_temporal=hf_vision.patch_temporal,
        merge_size=hf_vision.merge_size,
        pos_emb_height=hf_vision.pos_emb_height,
        pos_emb_width=hf_vision.pos_emb_width,
        rope_theta=rope["rope_theta"],
        projector_hidden_size=huggingface_config.projector_hidden_size,
        # The reference's window is pos_emb_height patches on a side.
        window_size_patches=hf_vision.pos_emb_height,
    )
    if config.out_hidden_size != huggingface_config.out_hidden_size:
        raise ValueError(
            f"out_hidden_size {huggingface_config.out_hidden_size} does"
            f" not match the merged width {config.out_hidden_size}"
        )
    return config
