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
"""Registration of MiniMax Music 3 as an audio generation architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.graph.weights import WeightsFormat
from max.pipelines.context import AudioContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig
from typing_extensions import Self

from .music3_executor import MiniMaxMusic3Executor
from .tokenizer import MAX_PROMPT_TOKENS, MiniMaxMusic3Tokenizer


@dataclass(kw_only=True)
class MiniMaxMusic3ArchConfig(ArchConfig):
    """Pipeline-level config. No KV cache here: the autoregressive stage owns
    its own cache manager, because its positions are frames rather than the
    request's tokens."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    # Only bfloat16 was gated against the reference, and it is also the only
    # encoding whose weights fit: float32 is 47 GiB.
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    pipeline_config: PipelineConfig
    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        """The prompt's ceiling, which is the tokenizer's.

        Not the autoregressive model's 10240 positions: those are shared
        between the prompt and the frames it generates, so the executor -- not
        the tokenizer -- is where that budget is divided.
        """
        return MAX_PROMPT_TOKENS

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The prompt's ceiling, which is model metadata rather than a
        deployment length: see :meth:`get_max_seq_len`."""
        del huggingface_config, model_config
        return MAX_PROMPT_TOKENS

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Validates that this checkpoint can run at all.

        ``max_seq_len`` is ignored: this model's sequence length is the
        tokenizer's prompt ceiling rather than a length the deployment can
        clamp.

        Raises:
            ValueError: If the manifest has no ``transformer`` component, or if
                more than one device was requested.
        """
        del max_seq_len
        if model_config is None:
            model_config = pipeline_config.models.get("transformer")
        if model_config is None:
            raise ValueError(
                "MiniMax Music 3 requires a 'transformer' component in the "
                "model manifest."
            )
        if len(model_config.device_specs) != 1:
            raise ValueError(
                "MiniMax Music 3 is single-device only: its stages take turns "
                "on one GPU rather than sharding across several."
            )
        return cls(pipeline_config=pipeline_config)


minimax_music3_arch = SupportedArchitecture(
    name="MiniMaxMusic3ModularPipeline",
    task=PipelineTask.AUDIO_GENERATION,
    default_encoding=MiniMaxMusic3ArchConfig.DEFAULT_ENCODING,
    supported_encodings=MiniMaxMusic3ArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=["MiniMaxAI/MiniMax-Music3"],
    pipeline_model=MiniMaxMusic3Executor,
    context_type=AudioContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=MiniMaxMusic3Tokenizer,
    config=MiniMaxMusic3ArchConfig,
)
