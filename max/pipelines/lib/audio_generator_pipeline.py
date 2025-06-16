# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Audio generator pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from max.nn import ReturnLogits
from max.pipelines.core import (
    AudioGenerationResponse,
    AudioGenerator,
    TTSContext,
)

if TYPE_CHECKING:
    from .config import PipelineConfig
from .pipeline import PipelineModel


class AudioGeneratorPipeline(AudioGenerator[TTSContext]):
    """Converts text to speech.

    This pipeline passes all of the work through to the PipelineModel.
    """

    @no_type_check
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel],
        **unused_kwargs,
    ) -> None:
        """Initializes the TTS pipeline.

        Args:
            pipeline_config: The configuration for the pipeline.
            pipeline_model: The pipeline model to use.
        """
        # Create the pipeline model.
        # None of the arguments are used except for the config.
        self.pipeline_model = pipeline_model(
            pipeline_config=pipeline_config,
            session=None,
            huggingface_config=None,
            encoding=None,
            devices=None,
            kv_cache_config=None,
            weights=None,
            adapter=None,
            return_logits=ReturnLogits.ALL,
        )
        assert hasattr(self.pipeline_model, "speech_lm_pipeline")
        self.speech_lm_pipeline = self.pipeline_model.speech_lm_pipeline

    def next_chunk(
        self, batch: dict[str, TTSContext], num_tokens: int
    ) -> dict[str, AudioGenerationResponse]:
        next_chunk = getattr(self.pipeline_model, "next_chunk")  # type: ignore[has-type]
        return next_chunk(batch, num_tokens)

    def release(self, context: TTSContext) -> None:
        release = getattr(self.pipeline_model, "release")  # type: ignore[has-type]
        release(context)

    @property
    def decoder_sample_rate(self) -> int:
        return getattr(self.pipeline_model, "decoder_sample_rate")  # type: ignore[has-type]
