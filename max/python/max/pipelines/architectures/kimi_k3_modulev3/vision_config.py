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
"""Configuration for the Kimi K3 image tower."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import PipelineConfig
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import Self


def has_vision_tower(huggingface_config: PretrainedConfig) -> bool:
    """Returns whether the checkpoint configures an image tower."""
    # The registered HF config shim turns an absent or null `vision_config`
    # into an empty object, so test for the geometry.
    hf_vision_config = getattr(huggingface_config, "vision_config", None)
    return getattr(hf_vision_config, "patch_size", None) is not None


@dataclass(kw_only=True)
class KimiK3VisionConfig:
    """Holds the checkpoint's image tower dimensions and placement."""

    dtype: DType
    devices: list[DeviceRef]
    patch_size: int
    vt_hidden_size: int
    qkv_hidden_size: int
    vt_intermediate_size: int
    vt_num_attention_heads: int
    vt_num_hidden_layers: int
    init_pos_emb_height: int
    init_pos_emb_width: int
    init_pos_emb_time: int
    merge_kernel_size: tuple[int, int]
    mm_hidden_size: int
    text_hidden_size: int
    projector_ln_eps: float
    tower_norm_eps: float = 1.1920928955078125e-7
    in_channels: int = 3
    rope_max_height: int = 512
    rope_max_width: int = 512
    rope_theta: float = 10000.0

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: PretrainedConfig,
    ) -> Self | None:
        """Reads the vision geometry from the checkpoint configuration.

        Returns:
            The tower's configuration, or ``None`` for a text-only checkpoint.
        """
        if not has_vision_tower(huggingface_config):
            return None
        hf_vision_config = huggingface_config.vision_config
        return cls(
            dtype=DType.bfloat16,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            patch_size=hf_vision_config.patch_size,
            vt_hidden_size=hf_vision_config.vt_hidden_size,
            qkv_hidden_size=hf_vision_config.qkv_hidden_size,
            vt_intermediate_size=hf_vision_config.vt_intermediate_size,
            vt_num_attention_heads=hf_vision_config.vt_num_attention_heads,
            vt_num_hidden_layers=hf_vision_config.vt_num_hidden_layers,
            init_pos_emb_height=hf_vision_config.init_pos_emb_height,
            init_pos_emb_width=hf_vision_config.init_pos_emb_width,
            init_pos_emb_time=hf_vision_config.init_pos_emb_time,
            merge_kernel_size=tuple(hf_vision_config.merge_kernel_size),
            mm_hidden_size=hf_vision_config.mm_hidden_size,
            text_hidden_size=hf_vision_config.text_hidden_size,
            projector_ln_eps=hf_vision_config.projector_ln_eps,
        )
