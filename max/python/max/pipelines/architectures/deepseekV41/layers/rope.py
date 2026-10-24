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
"""Per-layer rotary position embedding for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from max.graph import DeviceRef
from max.nn.rotary_embedding import YarnScalingParams
from transformers import AutoConfig

from ...deepseekV4.layers.rope import DeepseekV4RotaryEmbedding
from ..model_config import DeepseekV41Config

__all__ = ["rope_for_layer"]


def rope_for_layer(
    config: AutoConfig | DeepseekV41Config,
    layer_idx: int,
    max_seq_len: int,
    device: DeviceRef,
) -> DeepseekV4RotaryEmbedding:
    """Builds the rotary table shared by a layer's attention and compressor.

    Rotation is interleaved (adjacent element pairs). Layers that compress
    their KV (``compress_ratios[layer_idx] > 0``) use ``compress_rope_theta``
    with YaRN frequency scaling; the others use ``rope_theta`` unscaled.

    Args:
        config: The raw HF text config or the flattened
            ``DeepseekV41Config``; both carry every field read here.
        layer_idx: Index into ``config.compress_ratios``, which lists the
            decoder layers followed by the DSpark stages.
        max_seq_len: Exclusive upper bound on positions gathered from the
            returned table's ``freqs_cis_base()``.
        device: Device the ``freqs_cis_base()`` table is placed on.

    Returns:
        The layer's :class:`DeepseekV4RotaryEmbedding`.
    """
    compressed = config.compress_ratios[layer_idx] > 0
    scaling = config.rope_scaling or {}
    rope = DeepseekV4RotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=config.compress_rope_theta if compressed else config.rope_theta,
        max_seq_len=max_seq_len,
        head_dim=config.qk_rope_head_dim,
        interleaved=True,
        scaling_params=YarnScalingParams(
            factor=float(scaling["factor"]),
            beta_fast=float(scaling["beta_fast"]),
            beta_slow=float(scaling["beta_slow"]),
            original_max_position_embeddings=int(
                scaling["original_max_position_embeddings"]
            ),
            # Unused by the YaRN frequency path.
            truncate=False,
        )
        if compressed
        else None,
    )
    rope.device = device
    return rope
