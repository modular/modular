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
"""DeepSeek-V4's DSpark stages: their config and module builder.

The stages run only beside their trunk, inside a unified spec-decode graph.
The config builds the stages' module for that graph from the trunk's own
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

from ..deepseekV4.deepseekV4 import DeepseekV4DSpark
from ..deepseekV4.model import set_checkpoint_dtypes
from ..deepseekV4.model_config import DeepseekV4Config

__all__ = [
    "DSparkDeepseekV4DraftArchConfig",
    "DSparkDeepseekV4DraftConfig",
    "dspark_draft_width",
]


def dspark_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """The DSpark block width: the checkpoint's ``dspark_block_size``.

    Every block position drafts a token, so the width is the block size. The
    stages unroll the Markov chain over the block, so a different
    ``--num-speculative-tokens`` is rejected rather than silently overridden.
    """
    del draft_huggingface_config
    block_size = int(target_huggingface_config.dspark_block_size)
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != block_size:
        # TODO(dsv4-dspark W4): take the block width at run time.
        raise ValueError(
            f"DeepSeek-V4 DSpark drafts dspark_block_size={block_size} tokens;"
            f" --num-speculative-tokens={requested} is not supported yet"
        )
    return block_size


@dataclass
class DSparkDeepseekV4DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
    """The draft architecture's config: its sequence bound, and no KV leaf.

    The stages' windows are layers of the trunk's window leaf, which the
    trunk's own KV params already size.
    """

    max_seq_len: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> DSparkDeepseekV4DraftArchConfig:
        """Records the sequence length the pipeline resolved."""
        del pipeline_config, model_config
        return cls(max_seq_len=max_seq_len)

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> None:
        """Returns ``None``: the stages read and write the trunk's leaves."""
        del huggingface_config, pipeline_config, devices
        del kv_cache_config, cache_dtype

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV4DSpark, DSparkDeepseekV4DraftConfig]:
        """Builds the stages from the trunk's checkpoint, which ``context``
        names.

        The stages read the trunk's embedding and head, which their caller
        passes in, so the checkpoint's ``mtp.*`` weights are all they load.

        Raises:
            ValueError: If the method isn't ``dspark``.
        """
        speculative = context.pipeline_config.speculative
        assert speculative is not None
        if speculative.speculative_method != "dspark":
            raise ValueError(
                "DeepSeek-V4's DSpark stages require --speculative-method"
                f" dspark, got {speculative.speculative_method!r}."
            )
        config = DeepseekV4Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        set_checkpoint_dtypes(
            config,
            context.huggingface_config,
            # The trunk probes its final norm and its embedding; the stages'
            # norms take the same rule, and the embedding passes through
            # unconverted.
            norm_dtype=state_dict["mtp.0.attn_norm.weight"].dtype,
            embed_dtype=context.weights["embed.weight"].data().dtype,
        )
        draft = DeepseekV4DSpark(config)
        draft.load_state_dict(state_dict, weight_alignment=1, strict=True)
        return draft, DSparkDeepseekV4DraftConfig(config)


@dataclass(frozen=True)
class DSparkDeepseekV4DraftConfig:
    """The stages' module config, and what the pairing reads beside it."""

    module: DeepseekV4Config

    @property
    def hidden_size(self) -> int:
        """Returns the trunk's hidden size, which the stages draft in."""
        return self.module.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the trunk's vocabulary, which the stages score."""
        return self.module.vocab_size

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the trunk layers whose states the stages read."""
        return self.module.dspark_target_layer_ids

    @property
    def mask_token_id(self) -> int:
        """Returns the token that fills the block past its anchor."""
        return self.module.dspark_noise_token_id

    @property
    def block_size(self) -> int:
        """Returns the tokens a block holds, every one of them drafted."""
        return self.module.dspark_block_size
