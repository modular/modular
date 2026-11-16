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
"""Weight loading for the Kimi K3 DSpark draft.

The draft module tree mirrors its checkpoint's names, so tensors pass through
as stored. The only tensor nothing reads is the confidence head's: it gates
adaptive proposal lengths, which MAX's block driver does not consume, so the
adapter drops it rather than loading a module nothing calls.
"""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData

from .config import K3DSparkDraftConfig

__all__ = [
    "SKIPPED_DRAFT_PREFIXES",
    "convert_k3_dspark_state_dict",
    "expected_k3_dspark_weight_names",
]

SKIPPED_DRAFT_PREFIXES = ("confidence_head.",)
"""Draft checkpoint weights nothing in this pipeline reads."""


def expected_k3_dspark_weight_names(
    config: K3DSparkDraftConfig,
) -> set[str]:
    """Returns the checkpoint tensor names the draft module reads.

    The names are those the module's attributes spell, so this is the
    contract between the checkpoint and :class:`~.dspark.K3DSpark`.
    """
    names: set[str] = {
        "context_proj.weight",
        "context_norm.weight",
        "embed_tokens.weight",
        "final_norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
        attn = f"{prefix}.self_attn"
        for proj in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ):
            names.add(f"{attn}.{proj}.weight")
        names.add(f"{attn}.q_a_layernorm.weight")
        names.add(f"{attn}.kv_a_layernorm.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"{prefix}.mlp.{proj}.weight")
    return names


def convert_k3_dspark_state_dict(
    state_dict: Mapping[str, WeightData],
    config: K3DSparkDraftConfig,
) -> dict[str, WeightData]:
    """Returns the draft tensors its module reads.

    Raises:
        ValueError: If the checkpoint lacks or adds tensors the module does
            not name.
    """
    weights: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        if name.startswith(SKIPPED_DRAFT_PREFIXES):
            continue
        weights[name] = value
    expected = expected_k3_dspark_weight_names(config)
    missing, extra = expected - weights.keys(), weights.keys() - expected
    if missing or extra:
        raise ValueError(
            "The Kimi K3 DSpark checkpoint does not match its module. Missing:"
            f" {sorted(missing)[:5]}; unexpected: {sorted(extra)[:5]}."
        )
    return weights
