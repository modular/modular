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
"""Audio generator pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import load_devices
from max.interfaces import (
    AudioGenerationInputs,
    AudioGenerationOutput,
    Pipeline,
    RequestID,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TTSContext

if TYPE_CHECKING:
    from .config import PipelineConfig

from max.serve.telemetry.metrics import METRICS

from .interfaces import PipelineModel

AudioGeneratorPipelineType = Pipeline[
    AudioGenerationInputs[TTSContext], AudioGenerationOutput
]


class AudioGeneratorPipeline(AudioGeneratorPipelineType):
    """Converts text to speech.

    This pipeline passes all of the work through to the PipelineModel.

    Args:
        pipeline_config: The configuration for the pipeline.
        pipeline_model: The pipeline model to use.
        **unused_kwargs: Optional keyword arguments for API compatibility;
            ignored.
    """

    pipeline_model: PipelineModel[TTSContext]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TTSContext]],
        **unused_kwargs,
    ) -> None:
        # Create the pipeline model.
        # None of the arguments are used except for the config and devices.
        devices = load_devices(pipeline_config.model.device_specs)
        self.pipeline_model = pipeline_model(
            pipeline_config=pipeline_config,
            session=None,  # type: ignore
            devices=devices,
            kv_cache_config=None,  # type: ignore
            weights=None,  # type: ignore
            adapter=None,
            return_logits=ReturnLogits.ALL,
        )
        assert hasattr(self.pipeline_model, "speech_lm_pipeline")
        self.speech_lm_pipeline = self.pipeline_model.speech_lm_pipeline

    def execute(
        self, inputs: AudioGenerationInputs[TTSContext]
    ) -> dict[RequestID, AudioGenerationOutput]:
        """Runs the audio generation pipeline for the given inputs."""
        METRICS.input_tokens(
            sum(ctx.tokens.active_length for ctx in inputs.batch.values())
        )

        next_chunk = getattr(self.pipeline_model, "next_chunk")  # noqa: B009
        outputs = next_chunk(inputs.batch)
        METRICS.output_tokens(
            sum(output.steps_executed for output in outputs.values())
        )

        if hasattr(self.pipeline_model, "tts_config"):
            sample_rate = self.pipeline_model.tts_config.decoder_sample_rate
            for output in outputs.values():
                METRICS.audio_output_length(
                    int(output.audio_data.shape[0] / float(sample_rate) * 1000)
                )

        return outputs

    def release(self, request_id: RequestID) -> None:
        """Releases resources associated with the given request."""
        release = getattr(self.pipeline_model, "release")  # noqa: B009
        release(request_id)

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        assert hasattr(self.pipeline_model, "kv_manager")
        return self.pipeline_model.kv_manager
