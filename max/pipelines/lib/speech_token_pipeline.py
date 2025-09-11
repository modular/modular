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
"""Speech token generation pipeline for TTS model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from max.driver import DeviceStream
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    BatchLogitsProcessor,
    GenerationStatus,
    LogProbabilities,
    PipelineTokenizer,
    TextGenerationOutput,
)
from max.nn.kv_cache import KVCacheInputsSequence
from max.pipelines.core import TTSContext
from max.profiler import Tracer, traced

from .pipeline import PipelineModel, TextGenerationPipeline
from .sampling import FusedSamplingProcessor, apply_logits_processors

if TYPE_CHECKING:
    from .config import PipelineConfig


class SpeechTokenGenerationPipeline(TextGenerationPipeline[TTSContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TTSContext]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> None:
        super().__init__(
            pipeline_config,
            pipeline_model,
            eos_token_id,
            weight_adapters,
            tokenizer,
        )
        self.d2h_stream = DeviceStream(self._devices[0])

    @traced
    def next_speech_token(
        self,
        batch: dict[str, TTSContext],
        num_steps: int,
        tokens_to_generate: dict[str, int],
    ) -> dict[str, TextGenerationOutput]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        if not batch or num_steps == 0:
            return {}
        tracer: Tracer = Tracer("compute_parameters")

        batch = self._maybe_sort_loras(batch)

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())
        eos_token_list = list(self._eos_token_id)

        # Prepare the batch.
        model_inputs, num_steps, bitmask = self.prepare_batch(
            [context_batch], num_steps
        )

        # Multistep execution loop.
        tracer.next("prepare_sampling_processor")
        sampling_processor = FusedSamplingProcessor(
            sampler=self._sampler,
            pipeline_config=self._pipeline_config,
            context_batch=context_batch,
            num_steps=num_steps,
            device=self._devices[0],
            bitmask=bitmask,
            vocab_size=self.vocab_size,
        )
        batch_processors: list[BatchLogitsProcessor] = [sampling_processor]

        curr_step_inputs = model_inputs

        seq_has_eos = np.zeros([len(context_batch)], dtype=np.bool)
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")

            # Execute the model and get next tokens.
            model_outputs = self._pipeline_model.execute(
                model_inputs=curr_step_inputs
            )

            if i > 0:
                new_tokens_np = new_tokens.to_numpy()  # type: ignore
                seq_has_eos |= np.isin(new_tokens_np, eos_token_list)

            # Sample next token.
            tracer.next("sample_next_token")
            apply_logits_processors(
                context_batch=context_batch,
                batch_logits=model_outputs.logits,
                batch_logit_offsets=model_outputs.logit_offsets,
                batch_processors=batch_processors,
            )
            new_tokens = sampling_processor.new_tokens
            assert new_tokens is not None

            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1 or seq_has_eos.all():
                tracer.pop()  # pops f"step_{i}"
                break
            # Prepare inputs for the next token in multistep execution
            tracer.next("increment_cache_lengths")  # pops sample_next_token

            assert isinstance(
                curr_step_inputs.kv_cache_inputs, KVCacheInputsSequence
            ), (
                "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
            )
            assert isinstance(
                curr_step_inputs.kv_cache_inputs.kv_cache_inputs, list
            ), "increment_cache_lengths instantiates and passes this as a list"
            curr_step_inputs.kv_cache_inputs.kv_cache_inputs = (
                self._pipeline_model.kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            tracer.next("prepare_next_token_inputs")  # pops inc_cache_lengths
            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
            new_tokens = new_tokens.to(self.d2h_stream)
            tracer.pop()  # pops step_{i}

        # Do the copy to host for each token generated.
        tracer.next(
            "generated_tokens.to(CPU())"
        )  # pops multistep_execution_loop_steps
        generated_tokens_host = sampling_processor.generated_tokens.to_numpy()

        num_steps = i + 1

        # Prepare the response, pruning away completed requests as we go.
        res: dict[str, TextGenerationOutput] = {}
        tracer.push("prepare_response")
        for batch_index, (request_id, context) in enumerate(batch.items()):
            status = GenerationStatus.ACTIVE
            start_log_probs: Optional[list[LogProbabilities]] = None
            if context.log_probabilities:
                start_log_probs = []

            num_valid_tokens = min(num_steps, tokens_to_generate[request_id])
            for step in range(num_valid_tokens):
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                context.update(new_token=next_token)

                if context.status.is_done:
                    break

            res[request_id] = context.to_generation_output()

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        tracer.next("kv_manager.step")  # pops prepare_response
        self._pipeline_model.kv_manager.step(context_batch)
        tracer.pop()  # pops kv_manager.step

        return res
