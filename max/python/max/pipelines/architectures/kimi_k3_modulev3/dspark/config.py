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
"""Config parsing for the Kimi-K3 DSpark draft checkpoint.

The published checkpoint (``model_type: k3_dspark``) is a
DSpark drafter: five dense layers of MLA attention with yarn RoPE, reading
five taps of its target's residual stream. The parser refuses every config
field this implementation does not build, so a checkpoint variant silently
unsupported here fails at config time instead of at the first draft step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

__all__ = [
    "K3_BLOCK_SIZE",
    "K3DSparkDraftConfig",
    "k3_dspark_draft_width",
    "parse_k3_dspark_config",
]

K3_BLOCK_SIZE = 7
"""The block width the published Kimi-K3 DSpark checkpoint was trained at.

Every block slot predicts -- DSpark's DeepSpec supervision, not DFlash's
masked-anchor one -- so a block of 7 positions ([anchor, mask x 6]) drafts 7
tokens and ``num_speculative_tokens`` is 7. The checkpoint's config.json
names no width; the drafter was trained and served at 7.
"""


@dataclass(frozen=True)
class K3DSparkDraftConfig:
    """The DSpark draft's hyperparameters, from its checkpoint config."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    vocab_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    """The target layers whose outputs the draft taps, post-layer (no shift)."""
    num_target_layers: int
    """The target depth the drafter was trained against (93 for Kimi K3).

    Read from the checkpoint's ``target_num_hidden_layers``, not its
    ``num_target_layers`` key (which counts the taps); the spec-decode
    pairing gate compares it with the served target's depth, and the
    parser bounds ``target_layer_ids`` by it.
    """
    markov_rank: int
    rms_norm_eps: float
    rope_theta: float
    rope_factor: float
    rope_original_max_position_embeddings: int
    rope_beta_fast: float
    rope_beta_slow: float
    rope_mscale: float
    rope_mscale_all_dim: float
    block_size: int = K3_BLOCK_SIZE

    @property
    def q_head_dim(self) -> int:
        """The per-head query width: nope plus rope."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def num_context_features(self) -> int:
        """How many target taps the ``context_proj`` consumes."""
        return len(self.target_layer_ids)

    @property
    def attention_scale(self) -> float:
        """The softmax scale: YaRN mscale squared over ``sqrt(q_head_dim)``.

        The drafter's cos/sin tables carry no mscale (the checkpoint declares
        ``mscale=1.0`` and ``mscale_all_dim=1.0``, whose ratio is 1.0), so
        the whole correction rides the scale, as DeepSeek's MLA does.
        """
        mscale = yarn_get_mscale(self.rope_factor, self.rope_mscale_all_dim)
        return (mscale * mscale) / math.sqrt(self.q_head_dim)


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """Returns the YaRN attention-magnitude correction."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _mapping(
    hf_config: AutoConfig | Mapping[str, object],
) -> Mapping[str, object]:
    """Returns the config's fields, whether passed as a mapping or not."""
    if isinstance(hf_config, Mapping):
        return hf_config
    return hf_config.to_dict()


def _int(config: Mapping[str, object], key: str) -> int:
    if key not in config:
        raise ValueError(f"k3_dspark config is missing required key {key!r}.")
    value = config[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"k3_dspark config {key!r}: expected int, got {value!r}"
        )
    return value


def _int_or(config: Mapping[str, object], key: str, default: int) -> int:
    return _int(config, key) if key in config else default


def _float_or(config: Mapping[str, object], key: str, default: float) -> float:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"k3_dspark config {key!r}: expected number, got {value!r}"
        )
    return float(value)


def _bool_or(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"k3_dspark config {key!r}: expected bool, got {value!r}"
        )
    return value


def _int_tuple(config: Mapping[str, object], key: str) -> tuple[int, ...]:
    if key not in config:
        raise ValueError(f"k3_dspark config is missing required key {key!r}.")
    value = config[key]
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(
            f"k3_dspark config {key!r}: expected ints, got {value!r}"
        )
    return tuple(value)


def _rope(config: Mapping[str, object]) -> Mapping[str, object]:
    """Returns the rope parameters, from either name a config gives them under.

    A config that carries ``rope_parameters: null`` (as a normalized
    config.json can) falls back to ``rope_scaling``, as the reference does.
    """
    value = config.get("rope_parameters")
    if value is None:
        value = config.get("rope_scaling")
    if not isinstance(value, Mapping):
        return {}
    return value


def parse_k3_dspark_config(
    hf_config: AutoConfig | Mapping[str, object],
) -> K3DSparkDraftConfig:
    """Reads the ``k3_dspark`` checkpoint config.

    Raises:
        ValueError: If the config describes a drafter this implementation does
            not build.
    """
    config = _mapping(hf_config)
    if config.get("model_type") != "k3_dspark":
        raise ValueError(
            "The Kimi K3 DSpark draft requires model_type k3_dspark, got"
            f" {config.get('model_type')!r}."
        )
    if config.get("markov_head_type", "vanilla") != "vanilla":
        raise ValueError(
            "The Kimi K3 DSpark draft's Markov head must be 'vanilla', got"
            f" {config.get('markov_head_type')!r}."
        )
    if not _int_or(config, "markov_rank", 0):
        raise ValueError(
            "The Kimi K3 DSpark draft config declares no markov_rank."
        )
    if _bool_or(config, "mla_use_output_gate", False):
        raise ValueError(
            "The Kimi K3 DSpark draft has no output gate; the published"
            " checkpoint carries no gate weights."
        )
    if _bool_or(config, "mla_use_nope", False):
        raise ValueError(
            "The Kimi K3 DSpark draft applies yarn RoPE to its rope slice."
        )
    rope = _rope(config)
    if rope.get("rope_type", rope.get("type")) != "yarn":
        raise ValueError(
            "The Kimi K3 DSpark draft requires yarn rope_scaling, got"
            f" {rope!r}."
        )
    block_size = _int_or(config, "block_size", K3_BLOCK_SIZE)
    if block_size != K3_BLOCK_SIZE:
        raise ValueError(
            "The Kimi K3 DSpark draft was trained at block_size"
            f" {K3_BLOCK_SIZE}; got {block_size}."
        )
    hidden_size = _int(config, "hidden_size")
    target_layer_ids = _int_tuple(config, "target_layer_ids")
    target_depth = _int(config, "target_num_hidden_layers")
    if any(not 0 <= layer < target_depth for layer in target_layer_ids):
        raise ValueError(
            "The Kimi K3 DSpark draft's target_layer_ids must index the"
            f" {target_depth} layers of the target it declares"
            f" (target_num_hidden_layers); got {list(target_layer_ids)}."
        )
    # The reference sizes context_proj's input at num_target_layers *
    # target_hidden_size, while the module concatenates the tapped rows
    # at the draft's own width: len(target_layer_ids) * hidden_size. The
    # published checkpoint's values agree, so refuse the variants whose
    # would not, here rather than at the first weight load.
    target_hidden_size = _int_or(config, "target_hidden_size", hidden_size)
    if target_hidden_size != hidden_size:
        raise ValueError(
            "The Kimi K3 DSpark draft projects its target's taps at its"
            f" own hidden_size {hidden_size}, but the checkpoint declares"
            f" target_hidden_size {target_hidden_size}."
        )
    trained_taps = _int(config, "num_target_layers")
    if trained_taps != len(target_layer_ids):
        raise ValueError(
            "The Kimi K3 DSpark draft reads len(target_layer_ids)="
            f"{len(target_layer_ids)} taps, but the checkpoint declares"
            f" num_target_layers={trained_taps}."
        )
    return K3DSparkDraftConfig(
        hidden_size=hidden_size,
        intermediate_size=_int(config, "intermediate_size"),
        num_hidden_layers=_int(config, "num_hidden_layers"),
        num_attention_heads=_int(config, "num_attention_heads"),
        q_lora_rank=_int(config, "q_lora_rank"),
        kv_lora_rank=_int(config, "kv_lora_rank"),
        qk_nope_head_dim=_int(config, "qk_nope_head_dim"),
        qk_rope_head_dim=_int(config, "qk_rope_head_dim"),
        v_head_dim=_int(config, "v_head_dim"),
        vocab_size=_int(config, "vocab_size"),
        mask_token_id=_int(config, "mask_token_id"),
        target_layer_ids=target_layer_ids,
        num_target_layers=target_depth,
        markov_rank=_int(config, "markov_rank"),
        rms_norm_eps=_float_or(config, "rms_norm_eps", 1e-5),
        # The reference resolves the base top-level first, then the rope
        # dict, then its own 10000.0 default; follow it exactly.
        rope_theta=_float_or(
            config, "rope_theta", _float_or(rope, "rope_theta", 10000.0)
        ),
        rope_factor=_float_or(rope, "factor", 1.0),
        rope_original_max_position_embeddings=_int_or(
            rope, "original_max_position_embeddings", 32768
        ),
        rope_beta_fast=_float_or(rope, "beta_fast", 32.0),
        rope_beta_slow=_float_or(rope, "beta_slow", 1.0),
        rope_mscale=_float_or(rope, "mscale", 1.0),
        # 0.0 when a yarn config omits it, deliberately not 1.0: at 0.0 the
        # mscale correction rides the cos/sin tables and the scale falls to
        # 1/sqrt(q_head_dim).
        rope_mscale_all_dim=_float_or(rope, "mscale_all_dim", 0.0),
        block_size=block_size,
    )


def k3_dspark_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: AutoConfig | None,
    draft_huggingface_config: AutoConfig | Mapping[str, object] | None,
) -> int:
    """Returns the draft width the checkpoint was trained for: its block size.

    Every block slot drafts, so the width equals the block size, 7.

    Raises:
        ValueError: If a different ``num_speculative_tokens`` is requested.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    block_size = _int_or(
        _mapping(draft_huggingface_config), "block_size", K3_BLOCK_SIZE
    )
    if block_size != K3_BLOCK_SIZE:
        raise ValueError(
            "The Kimi K3 DSpark draft was trained at block_size"
            f" {K3_BLOCK_SIZE}; the checkpoint declares {block_size}."
        )
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != block_size:
        raise ValueError(
            "The Kimi K3 DSpark draft drafts exactly block_size="
            f"{block_size} tokens per step (the width the checkpoint was"
            f" trained with); got num_speculative_tokens={requested}."
        )
    return block_size
