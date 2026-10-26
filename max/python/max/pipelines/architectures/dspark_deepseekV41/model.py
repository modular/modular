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
"""DeepSeek-V4.1's DSpark stages: their config and module builder.

The stages run only beside their trunk, inside a speculative graph. The
config builds the stages' module for that graph from the trunk's own
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..deepseekV41.dspark_drafter import DeepseekV41DSparkDrafter
from ..deepseekV41.model_config import DeepseekV41Config

__all__ = [
    "DSparkDeepseekV41DraftArchConfig",
    "DSparkDeepseekV41DraftConfig",
]


@dataclass
class DSparkDeepseekV41DraftArchConfig(ArchConfigWithBoundedMaxSeqLen):
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
    ) -> DSparkDeepseekV41DraftArchConfig:
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
    ) -> tuple[DeepseekV41DSparkDrafter, DSparkDeepseekV41DraftConfig]:
        """Builds the stages from the trunk's checkpoint, which ``context``
        names.

        The stages read the trunk's embedding and head, which their caller
        passes in, so the checkpoint's ``mtp.*`` weights are all they load.
        Their position tables cover the slack of the trunk's KV params,
        which ``context`` carries.

        Raises:
            ValueError: If the method isn't ``dspark``, or the checkpoint
                names no token to fill the block with.
        """
        speculative = context.pipeline_config.speculative
        assert speculative is not None
        if speculative.speculative_method != "dspark":
            raise ValueError(
                "DeepSeek-V4.1's DSpark stages require --speculative-method"
                f" dspark, got {speculative.speculative_method!r}."
            )
        config = DeepseekV41Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        if config.dspark_noise_token_id is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs dspark_noise_token_id to fill the"
                " draft block."
            )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParamInterface)
            config.kv_params = context.kv_params
        draft = DeepseekV41DSparkDrafter(config, config.devices[0])
        draft.load_state_dict(
            parse_state_dict_from_weights(
                context.pipeline_config, context.weights, context.adapter
            ),
            weight_alignment=1,
            strict=True,
        )
        return draft, DSparkDeepseekV41DraftConfig(config)


@dataclass(frozen=True)
class DSparkDeepseekV41DraftConfig:
    """The stages' module config, and what the pairing reads beside it."""

    module: DeepseekV41Config

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
        assert self.module.dspark_noise_token_id is not None
        return self.module.dspark_noise_token_id

    @property
    def block_size(self) -> int:
        """Returns the tokens a block holds, every one of them drafted."""
        return self.module.dspark_block_size
