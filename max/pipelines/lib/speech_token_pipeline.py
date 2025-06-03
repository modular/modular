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

import logging
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch
from max.driver import DeviceStream, Tensor
from max.dtype import DType
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.nn.kv_cache import KVCacheInputsSequence
from max.pipelines.core import (
    InputContext,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
)
from max.profiler import Tracer, traced

from .pipeline import (
    PipelineModel,
    TextGenerationPipeline,
)

if TYPE_CHECKING:
    from .config import PipelineConfig

T = TypeVar("T", bound=InputContext)

logger = logging.getLogger("max.pipelines")


class SpeechTokenGenerationPipeline(TextGenerationPipeline):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
    ) -> None:
        super().__init__(
            pipeline_config,
            pipeline_model,
            eos_token_id,
            weight_adapters,
        )
        self.d2h_stream = DeviceStream(self._devices[0])

    @traced
    def next_speech_token(
        self,
        batch: dict[str, T],
        num_steps: int,
        tokens_to_generate: dict[str, int],
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        tracer: Tracer = Tracer("compute_parameters")

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())
        eos_token_list = list(self._eos_token_id)

        # Prepare the batch.
        model_inputs, num_steps, bitmask = self.prepare_batch(
            context_batch, num_steps
        )

        # Multistep execution loop.
        tracer.next("allocate_generated_tokens")
        generated_tokens = Tensor(
            shape=(len(context_batch), 0),
            dtype=DType.int64,
            device=self._devices[0],
        )

        if self._pipeline_config.sampling_config.do_penalties:
            frequency_data = []

            # Only build penalty frequency data if frequency or presence penalties are actually used
            if (
                self._pipeline_config.sampling_config.frequency_penalty != 0
                or self._pipeline_config.sampling_config.presence_penalty != 0
            ):
                frequency_data.append(
                    self._build_token_frequency_csr(context_batch, num_steps)
                )

            # Only build repetition frequency data if repetition penalty is actually used
            if self._pipeline_config.sampling_config.repetition_penalty != 1:
                frequency_data.append(
                    self._build_token_frequency_csr(
                        context_batch, num_steps, include_prompt=True
                    )
                )
        else:
            frequency_data = None

        curr_step_inputs = model_inputs

        seq_has_eos = np.zeros([len(context_batch)], dtype=np.bool)
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")

            # Execute the model and get next tokens.
            model_outputs = self._pipeline_model.execute(
                model_inputs=curr_step_inputs,
            )

            if i > 0:
                new_tokens_np = new_tokens.to_numpy()  # type: ignore
                seq_has_eos |= np.isin(new_tokens_np, eos_token_list)

            if bitmask is not None:
                assert self.vocab_size is not None
                bits = 2 ** torch.arange(32, dtype=torch.int32)
                bitmask = (bitmask.unsqueeze(-1) & bits) != 0
                bitmask = bitmask.reshape(
                    len(context_batch),
                    -1,
                ).to(torch.bool)
                bitmask = bitmask[:, 0 : self.vocab_size]

                bitmask = Tensor.from_dlpack(bitmask).to(self._devices[0])

            # Sample next token.
            tracer.next("sample_next_token")
            new_tokens, new_generated_tokens = self.sample_logits(
                model_outputs.logits,
                generated_tokens,
                logit_offsets=model_outputs.logit_offsets,
                bitmask=bitmask,
                frequency_data=frequency_data,
            )

            assert isinstance(new_tokens, Tensor)
            assert isinstance(new_generated_tokens, Tensor)
            generated_tokens = new_generated_tokens

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
        generated_tokens_host = generated_tokens.to_numpy()

        num_steps = i + 1

        # Prepare the response, pruning away completed requests as we go.
        res: dict[str, TextGenerationResponse] = {}
        tracer.push("prepare_response")
        for batch_index, (request_id, context) in enumerate(batch.items()):
            status = TextGenerationStatus.ACTIVE
            res[request_id] = TextGenerationResponse([], status)
            num_valid_tokens = min(num_steps, tokens_to_generate[request_id])
            for step in range(num_valid_tokens):
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                context.update(
                    new_token=next_token,
                )

                res[request_id].update_status(context.status)
                if context.is_done:
                    break

            # Walk outstanding completion tokens, and return to user.
            for token, log_probs in context.outstanding_completion_tokens():
                res[request_id].append_token(TextResponse(token, log_probs))

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        tracer.next("kv_manager.step")  # pops prepare_response
        self._pipeline_model.kv_manager.step(context_batch)
        tracer.pop()  # pops kv_manager.step

        return res
