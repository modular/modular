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
"""The DSpark drafter an Inkling 2 checkpoint ships in ``mtp.safetensors``.

The drafter predicts a block of ``block_size`` tokens at once: an anchor (the
last committed token) followed by ``block_size - 1`` mask positions, each
predicting the token at its own position. Its layers attend bidirectionally
over the block and over context keys and values computed from target
features: the target's hidden states at ``target_layer_ids``. Each drafter
layer and the recurrent Markov head mixes those features with its own row of
``fuse_weight``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from typing_extensions import Self

MARKOV_HEAD_TYPE = "rnn_hidden"
"""The only Markov head the checkpoints ship."""


@dataclass(kw_only=True, frozen=True)
class Inkling2DSparkConfig:
    """The drafter's config, parsed from the checkpoint's ``mtp_config``."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float

    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    """Positions a local layer sees before the block."""

    block_size: int
    """Rows per drafted block: the anchor and ``block_size - 1`` masks."""

    markov_rank: int
    target_layer_ids: tuple[int, ...]
    """Target layers whose input the drafter reads, in layer order. The last
    id is the target's ``num_hidden_layers`` and stands for its final-norm
    output."""

    vocab_size: int
    """The target's padded vocabulary, which the Markov head embeds."""

    @property
    def num_speculative_tokens(self) -> int:
        """Tokens one drafted block proposes: one per mask position."""
        return self.block_size - 1

    @property
    def num_target_features(self) -> int:
        """How many target hidden states the drafter reads per token."""
        return len(self.target_layer_ids)

    @property
    def feature_width(self) -> int:
        """The width of one token's concatenated target features."""
        return self.num_target_features * self.hidden_size

    @property
    def tapped_layer_inputs(self) -> tuple[int, ...]:
        """The target layers whose input is captured, in layer order."""
        return self.target_layer_ids[:-1]

    @property
    def num_fuse_rows(self) -> int:
        """Rows of ``fuse_weight``: one per drafter layer, then the head."""
        return self.num_hidden_layers + 1

    def is_local_attention(self, layer_idx: int) -> bool:
        """Whether this drafter layer uses sliding-window attention."""
        return layer_idx in self.local_layer_ids

    def attention_window(self, layer_idx: int) -> int | None:
        """The keys one block sees on a layer.

        That is the block itself and, on a local layer, the window before it.
        ``None`` on a layer that sees the whole context.
        """
        if not self.is_local_attention(layer_idx):
            return None
        return self.sliding_window_size + self.block_size

    def kv_bytes_per_token(self, dtype: DType = DType.bfloat16) -> int:
        """Key and value bytes one position takes on one drafter layer."""
        return (
            2 * self.num_key_value_heads * self.head_dim * dtype.size_in_bytes
        )

    def feature_bytes(
        self, num_tokens: int, dtype: DType = DType.bfloat16
    ) -> int:
        """Bytes of the target features for ``num_tokens`` tokens."""
        return num_tokens * self.feature_width * dtype.size_in_bytes

    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of every tensor ``mtp.safetensors`` holds, by name."""
        hidden, rank = self.hidden_size, self.markov_rank
        q_width = self.num_attention_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        shapes: dict[str, tuple[int, ...]] = {
            "fuse_weight": (self.num_fuse_rows, self.num_target_features),
            "mask_embed": (hidden,),
            "norm.weight": (hidden,),
            "markov_head.dflare_norm.weight": (hidden,),
            "markov_head.markov_w1.weight": (self.vocab_size, rank),
            # The recurrence reads [state, prev token, hidden, f_rnn] and
            # writes three gates.
            "markov_head.rnn_proj.weight": (3 * rank, 2 * rank + 2 * hidden),
            "markov_head.rnn_proj.bias": (3 * rank,),
            "markov_head.hidden_out.weight": (hidden, rank),
            "markov_head.hidden_out.bias": (hidden,),
        }
        for i in range(self.num_hidden_layers):
            layer = f"layers.{i}"
            shapes |= {
                f"{layer}.dflare_norm.weight": (hidden,),
                f"{layer}.input_layernorm.weight": (hidden,),
                f"{layer}.post_attention_layernorm.weight": (hidden,),
                f"{layer}.self_attn.q_proj.weight": (q_width, hidden),
                f"{layer}.self_attn.k_proj.weight": (kv_width, hidden),
                f"{layer}.self_attn.v_proj.weight": (kv_width, hidden),
                f"{layer}.self_attn.k_proj_ctx.weight": (kv_width, hidden),
                f"{layer}.self_attn.v_proj_ctx.weight": (kv_width, hidden),
                f"{layer}.self_attn.o_proj.weight": (hidden, q_width),
                f"{layer}.self_attn.q_norm.weight": (self.head_dim,),
                f"{layer}.self_attn.k_norm.weight": (self.head_dim,),
                f"{layer}.self_attn.k_norm_ctx.weight": (self.head_dim,),
                f"{layer}.mlp.gate_proj.weight": (
                    self.intermediate_size,
                    hidden,
                ),
                f"{layer}.mlp.up_proj.weight": (
                    self.intermediate_size,
                    hidden,
                ),
                f"{layer}.mlp.down_proj.weight": (
                    hidden,
                    self.intermediate_size,
                ),
            }
        return shapes

    @classmethod
    def from_hf(cls, huggingface_config: Any) -> Self:
        """Reads the drafter from the checkpoint's top-level config.

        Raises:
            ValueError: If the checkpoint ships no drafter, or one this
                parser does not describe.
        """
        mtp_config = getattr(huggingface_config, "mtp_config", None)
        if not mtp_config:
            raise ValueError("The Inkling 2 checkpoint has no mtp_config.")
        if not isinstance(mtp_config, Mapping):
            mtp_config = mtp_config.to_dict()
        text_config = huggingface_config.text_config
        num_target_layers = text_config.num_hidden_layers
        targets = tuple(mtp_config.get("target_layer_ids", ()))
        if (
            len(targets) < 2
            or list(targets) != sorted(set(targets))
            # The taps Inkling2.run_layers accepts, so a drafter it cannot
            # read is refused here rather than at the first forward.
            or not set(targets) <= set(range(num_target_layers + 1))
            or targets[-1] != num_target_layers
        ):
            raise ValueError(
                "Inkling 2 mtp_config target_layer_ids must be sorted target"
                " layer ids followed by the target's num_hidden_layers"
                f" ({num_target_layers}), its final-norm output; got"
                f" {list(targets)}."
            )
        num_layers = mtp_config["num_hidden_layers"]
        local_layer_ids = tuple(mtp_config.get("local_layer_ids", ()))
        if not set(local_layer_ids) <= set(range(num_layers)):
            raise ValueError(
                f"Inkling 2 mtp_config local_layer_ids {list(local_layer_ids)}"
                f" must index its {num_layers} layers."
            )
        unsupported = {
            "hidden_size": mtp_config["hidden_size"] != text_config.hidden_size,
            "markov_head_type": (
                mtp_config.get("markov_head_type") != MARKOV_HEAD_TYPE
            ),
            "q_bias": bool(mtp_config.get("q_bias")),
            "o_bias": bool(mtp_config.get("o_bias")),
        }
        if any(unsupported.values()):
            fields = sorted(k for k, v in unsupported.items() if v)
            raise ValueError(f"Unsupported Inkling 2 mtp_config: {fields}.")
        return cls(
            hidden_size=mtp_config["hidden_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=mtp_config["num_attention_heads"],
            num_key_value_heads=mtp_config["num_key_value_heads"],
            head_dim=mtp_config["head_dim"],
            intermediate_size=mtp_config["intermediate_size"],
            rms_norm_eps=mtp_config["rms_norm_eps"],
            rope_theta=mtp_config["rope_theta"],
            local_layer_ids=local_layer_ids,
            sliding_window_size=mtp_config["sliding_window_size"],
            block_size=mtp_config["block_size"],
            markov_rank=mtp_config["markov_rank"],
            target_layer_ids=targets,
            vocab_size=text_config.vocab_size,
        )
