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
"""Config for the DFlash2 draft of a Qwen3.5 target.

The draft HF config is the v1 ``dflash_config`` shape plus four keys v1 never
had (``conv_kernel_size``, ``conv_group_size``, ``selector_rank``,
``selector_top_k``), so the v1 parser is reused for the shared keys and
extended here rather than widened in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config

from ..llama3.model_config import Llama3Config

__all__ = [
    "DFlash2Qwen3_5Config",
    "Dflash2DraftHFConfig",
    "dflash2_draft_width",
    "parse_dflash2_draft_hf_config",
]

DRAFT_SLIDING_WINDOW = 2048
"""Positions the drafter can attend to, from the checkpoint's config.

The draft KV leaf is bounded at this rather than at the target's context: the
drafter's mask is windowed, so nothing older is ever read, and the leaf's cost
becomes a per-slot constant instead of scaling with ``--max-length``.
"""


@dataclass(frozen=True)
class Dflash2DraftHFConfig:
    """The ``dflash_config`` block of a DFlash2 draft checkpoint."""

    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int
    num_target_layers: int | None
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int

    def draft_width(self, speculative: SpeculativeConfig) -> int:
        """Returns ``block_size - 1``, refusing a flag that disagrees.

        The v1 helper warns and overrides. DFlash2 hard-errors instead
        (design D3): the width is not recoverable from the MEF, so the export
        records it alongside and the engine checks its own
        ``--num-speculative-tokens`` against that at boot. An overridden width
        would compile a block-8 graph and declare it was built for something
        else, and the engine would then feed the graph at the width it was
        told.

        Raises:
            ValueError: If ``num_speculative_tokens`` is set and is not
                ``block_size - 1``.
        """
        expected = self.block_size - 1
        requested = speculative.num_speculative_tokens
        if requested is not None and requested != expected:
            raise ValueError(
                f"DFlash2 was trained at block_size={self.block_size}, so it"
                f" drafts exactly {expected} tokens per step, but"
                f" --num-speculative-tokens is {requested}. The block width is"
                " structural (the drafter's convolution zeroes its t-1 tap at"
                " block-local position 0 and the selector walks a fixed"
                " tensor), so this is a hard error rather than an override."
            )
        return expected


def _dflash_config_of(huggingface_config: Any) -> Any:
    cfg = (
        huggingface_config.get("dflash_config")
        if isinstance(huggingface_config, dict)
        else getattr(huggingface_config, "dflash_config", None)
    )
    if cfg is None:
        raise ValueError(
            "DFlash2 draft HF config is missing ``dflash_config``."
        )
    return cfg


def _required_int(dflash_config: Any, name: str) -> int:
    value = (
        dflash_config.get(name)
        if isinstance(dflash_config, dict)
        else getattr(dflash_config, name, None)
    )
    if value is None:
        raise ValueError(
            f"DFlash2 draft HF config is missing ``dflash_config.{name}``."
        )
    return int(value)


def parse_dflash2_draft_hf_config(
    huggingface_config: Any,
) -> Dflash2DraftHFConfig:
    """Reads the DFlash2 drafter's structural constants off its HF config.

    Raises:
        ValueError: If a required key is missing, if the checkpoint declares no
            ``block_size``, or if it asks for an embedding/logit rescale this
            graph does not implement.
    """
    shared = parse_dflash_draft_hf_config(huggingface_config)
    if shared.block_size is None:
        raise ValueError(
            "DFlash2 draft HF config declares no ``dflash_config.block_size``."
            " The block width is structural — the drafter's convolution zeroes"
            " its t-1 tap at block-local position 0 — so it cannot be supplied"
            " by a flag."
        )
    dflash_config = _dflash_config_of(huggingface_config)

    # The reference multiplies the borrowed embedding and the borrowed head's
    # logits by these. Both are 1.0 on the shipped checkpoint and this graph
    # applies neither, so a checkpoint that sets them would draft from a
    # differently scaled space with no symptom but lost acceptance.
    for name in ("input_embedding_scale", "output_multiplier"):
        scale = (
            dflash_config.get(name, 1.0)
            if isinstance(dflash_config, dict)
            else getattr(dflash_config, name, 1.0)
        )
        if float(scale) != 1.0:
            raise ValueError(
                f"DFlash2 draft declares dflash_config.{name}={scale}; this"
                " graph borrows the target's embedding and head unscaled and"
                " implements neither rescale."
            )

    return Dflash2DraftHFConfig(
        mask_token_id=shared.mask_token_id,
        target_layer_ids=list(shared.target_layer_ids),
        block_size=shared.block_size,
        num_target_layers=shared.num_target_layers,
        conv_kernel_size=_required_int(dflash_config, "conv_kernel_size"),
        conv_group_size=_required_int(dflash_config, "conv_group_size"),
        selector_rank=_required_int(dflash_config, "selector_rank"),
        selector_top_k=_required_int(dflash_config, "selector_top_k"),
    )


def dflash2_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the width the DFlash2 draft checkpoint was trained for.

    Registered as the architecture's ``checkpoint_draft_width`` so that
    ``PipelineConfig.from_args`` resolves an omitted
    ``--num-speculative-tokens``. Without it the pipeline config keeps
    ``None`` while the graph is built at the checkpoint's width, and whatever
    reads the config instead of the graph -- the scheduler's draft-token
    handoff among them -- sees zero.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash2 requires a draft model.")
    return parse_dflash2_draft_hf_config(draft_huggingface_config).draft_width(
        speculative
    )


@dataclass(frozen=True)
class DFlash2Qwen3_5Config:
    """The draft's Llama 3 config and the DFlash2 fields its checkpoint
    names."""

    llama: Llama3Config
    dflash: Dflash2DraftHFConfig

    @property
    def hidden_size(self) -> int:
        """Returns the draft's hidden size."""
        return self.llama.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the draft's vocabulary size."""
        return self.llama.vocab_size

    @property
    def mask_token_id(self) -> int:
        """Returns the token that fills the block's drafted slots."""
        return self.dflash.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the draft reads."""
        return self.dflash.target_layer_ids

    @property
    def block_size(self) -> int:
        """Returns the block the draft was trained at."""
        return self.dflash.block_size

    @property
    def num_target_layers(self) -> int | None:
        """Returns the target depth the draft was trained against, if its
        checkpoint names one."""
        return self.dflash.num_target_layers
