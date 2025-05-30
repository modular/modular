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

from __future__ import annotations

import asyncio
import logging

import numpy as np
import soundfile as sf
from max.pipelines import PIPELINE_REGISTRY, AudioGenerationConfig, PipelineTask
from max.pipelines.core import (
    AudioGenerationRequest,
    AudioGenerator,
    PipelineAudioTokenizer,
)

logger = logging.getLogger(__name__)


def synthesize_speech(
    config: AudioGenerationConfig,
    text_to_synthesize: str,
    voice: str | None,
    output: str,
):
    """Synthesize speech from text, and save the result to a file.

    This will also ask the user whether they would like to rerun the synthesis
    with a different voice or prompt.

    Args:
        text_to_synthesize: The text to synthesize.
        config: The configuration for the speech synthesis pipeline.
    """
    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
        config,
        task=PipelineTask.AUDIO_GENERATION,
        override_architecture=config.audio_decoder,
    )

    assert isinstance(pipeline, AudioGenerator)
    assert isinstance(tokenizer, PipelineAudioTokenizer)

    async def generate_speech(text_to_synthesize, voice) -> np.ndarray | None:
        request = AudioGenerationRequest(
            id="example",
            index=0,
            model=config.model_config.model_path,
            voice=voice or "",
            input=text_to_synthesize,
        )
        try:
            context = await tokenizer.new_context(request)
            logger.info("Generating speech tokens for '%s'", text_to_synthesize)
            batch = {request.id: context}
            decode_response = pipeline.next_chunk(
                batch, config.max_length or 1024
            )
            logger.info(
                "Decoded audio for %s tokens", context.speech_tokens.shape
            )
            if request.id in decode_response:
                if decode_response[request.id].has_audio_data:
                    wav = decode_response[request.id].audio_data.reshape(-1)
                else:
                    wav = None
            else:
                wav = None
        except Exception as e:
            logger.error("Ran into error: %s", e)
            return None
        finally:
            pipeline.release(context)
        return wav

    while True:
        wav = asyncio.run(generate_speech(text_to_synthesize, voice))
        if wav is not None:
            decoder_sample_rate = pipeline.decoder_sample_rate
            logger.info(
                "Generated %.2f audio seconds, writing to %s",
                wav.shape[0] / decoder_sample_rate,
                output,
            )
            sf.write(output, wav, decoder_sample_rate, format="WAV")
        else:
            logger.warning("No audio generated.")

        input_text = input(
            """Rerun text to speech?
\t'n' to exit
\t'v=VOICE' to choose a different voice
Or type another prompt: """
        )
        if input_text == "n":
            break
        if input_text.startswith("v="):
            voice = input_text[2:]
        else:
            text_to_synthesize = input_text
