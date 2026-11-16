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
"""Rules that map the DSpark Nemotron-H draft checkpoint onto its module.

The draft module tree mirrors its checkpoint's names and declares each
tensor as stored, quantized ones included, so no rule renames, relabels or
drops a weight. The one tensor the module reads differently is
``embed_tokens.weight``: it is a frozen copy of the target's table with the
mask token's row trained, and the graph takes that row at init while the
table stays on the host.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import WeightRule
from transformers import PretrainedConfig

from .dspark_qwen3 import DSparkQwen3DraftConfig, draft_nvfp4_modules

__all__ = ["dspark_qwen3_weight_rules", "expected_draft_weight_names"]

_QUANT_SUFFIXES = ("weight", "weight_scale", "weight_scale_2")


def dspark_qwen3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns no rules: the module takes the checkpoint's names as stored.

    Args:
        huggingface_config: The draft checkpoint's config, which no rule
            reads.
    """
    del huggingface_config
    return ()


def expected_draft_weight_names(
    config: DSparkQwen3DraftConfig,
) -> set[str]:
    """Returns the checkpoint tensor names the draft module reads.

    The names are those the module's attributes spell, so this is the
    contract between the checkpoint and :class:`~.dspark_qwen3.DSparkQwen3`.
    """
    quantized = draft_nvfp4_modules(config)
    names: set[str] = set()

    def linear(module: str) -> None:
        if module in quantized:
            names.update(f"{module}.{s}" for s in _QUANT_SUFFIXES)
        else:
            names.add(f"{module}.weight")

    linear("fc")
    names.add("embed_tokens.weight")
    names.add("hidden_norm.weight")
    names.add("norm.weight")
    names.add("markov_head.markov_w1.weight")
    linear("markov_head.markov_w2")
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
        attn = f"{prefix}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{attn}.{proj}.weight")
        names.add(f"{attn}.q_norm.weight")
        names.add(f"{attn}.k_norm.weight")
        names.add(f"{attn}.attention_sink_bias")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            linear(f"{prefix}.mlp.{proj}")
    return names
