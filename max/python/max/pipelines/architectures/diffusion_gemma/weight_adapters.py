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
"""Weight rules for the DiffusionGemma checkpoint.

The checkpoint stores the shared transformer weights once, under
``model.decoder.*`` (HF ties every encoder layer weight to its decoder twin,
so only the decoder side plus the encoder-only ``layer_scalar`` buffers,
vision tower, and ``embed_vision`` are serialized). ``lm_head`` is absent —
tied to ``model.decoder.embed_tokens``.

MoE experts are stored per expert (``experts.J.{gate,up,down}_proj``), each as
an NVFP4 quadruplet: packed ``weight`` (u8), ``weight_scale`` (fp8-e4m3 group
scales), ``weight_scale_2`` and ``input_scale`` (fp32), the per-expert fields
``DiffusionGemmaQuantizedMoE`` reads (``moe.py``, a ``QuantizedMoE``). The BF16
checkpoint stores them stacked, as ``DiffusionGemmaMoE`` reads them.

The encoder and the denoiser load from the same adapted dict under the same
FQNs, but each compiles as its own graph and holds its own device copy of the
text weights.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig


def diffusion_gemma_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules that map a DiffusionGemma checkpoint onto its modules.

    The decoder prefix stripped, the MoE renames, and the vision tower's.
    The encoder-only ``layer_scalar`` buffers and the audio and patch
    embedders are ignored here; the encoder and the denoiser read the same
    renamed text weights. The single-weight-set port assumes HF tied each
    encoder layer's weights to its decoder twin, as the checkpoint format
    specifies; the rules are name-only (they serialize into the artifact
    for the Rust loader), so the tie itself is not re-verified per load.

    Args:
        huggingface_config: The checkpoint's config, unused: the rewrites
            are constants of the architecture.
    """
    del huggingface_config
    return (
        # The text weights: the decoder prefix stripped first so the MoE
        # rules see bare ``layers.N...`` names.
        WeightRenaming(r"^model\.decoder\.", ""),
        WeightRenaming(
            r"\.router\.proj\.weight$", ".moe_block.gate.gate_score.weight"
        ),
        WeightRenaming(r"\.router\.scale$", ".moe_block.gate.scale"),
        WeightRenaming(
            r"\.router\.per_expert_scale$", ".moe_block.gate.per_expert_scale"
        ),
        WeightRenaming(
            r"\.pre_feedforward_layernorm_2\.weight$",
            ".moe_block.pre_expert_norm.weight",
        ),
        WeightRenaming(r"\.experts\.", ".moe_block.experts."),
        # The vision tower and its multimodal embedder. The ``.linear.``
        # strip flattens HF's ClippableLinear wrapper, which only vision
        # names carry; it runs before the prefix strips and is anchored to
        # them, so a future text FQN containing ``.linear.`` keeps its name.
        WeightRenaming(
            r"^(model\.encoder\.(?:embed_vision|vision_tower))\.((?:[^.]+\.)*?)linear\.",
            r"\1.\2",
        ),
        WeightRenaming(r"^model\.encoder\.vision_tower\.", ""),
        WeightRenaming(r"^model\.encoder\.embed_vision", "embed_vision"),
        # The encoder-only layer_scalar buffers: the graphs share the
        # decoder's weights, so they are dropped.
        Ignore(
            r"^model\.encoder\.language_model\.",
            reason="the encoder's tied weights, which the decoder's copy serves",
        ),
    )
