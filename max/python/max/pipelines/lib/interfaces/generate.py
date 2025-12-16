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
"""Interfaces for text generation and generation mixins."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..config import PipelineConfig

from max.interfaces import (
    GenerationStatus,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    RequestType,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.kv_cache import NullKVCacheManager
from max.kv_cache.paged_cache import PagedKVCacheManager


@runtime_checkable
class GenerateMixin(Protocol[TextGenerationContextType, RequestType]):
    @property
    def kv_managers(self) -> list[PagedKVCacheManager | NullKVCacheManager]: ...

    @property
    def pipeline_config(self) -> PipelineConfig: ...

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType, npt.NDArray[np.integer[Any]], RequestType
    ]: ...

    def execute(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> PipelineOutputsDict[TextGenerationOutput]: ...

    def release(self, request_id: RequestID) -> None: ...

    def generate(
        self, prompts: RequestType | list[RequestType]
    ) -> list[TextGenerationOutput]:
        """Generates outputs for the given prompts."""
        if not isinstance(prompts, list):
            prompts = [prompts]

        async def _generate() -> list[TextGenerationOutput]:
            res = [
                TextGenerationOutput(
                    request_id=prompt.request_id,
                    tokens=[],
                    final_status=GenerationStatus.ACTIVE,
                )
                for prompt in prompts
            ]
            async for outputs in self.generate_async(prompts):
                for i, output in enumerate(outputs):
                    res[i].tokens.extend(output.tokens)
                    if output.is_done:
                        res[i].final_status = output.final_status
            return res

        return asyncio.run(_generate())

    async def generate_async(
        self, prompts: RequestType | list[RequestType]
    ) -> Any:
        if not isinstance(prompts, list):
            prompts = [prompts]

        context_batch: list[TextGenerationContextType] = []
        for prompt in prompts:
            context = await self.tokenizer.new_context(prompt)
            context_batch.append(context)

        kv_managers = self.kv_managers
        data_parallel_degree = (
            self.pipeline_config.model_config.data_parallel_degree
        )

        # Create inputs to the model. If data parallelism is enabled, group them
        # by replica.
        batches: list[dict[RequestID, TextGenerationContextType]] = []
        batch_to_replica_idx: dict[RequestID, int] = {}
        if data_parallel_degree > 1 and len(kv_managers) > 1:
            # We don't support speculative decoding when data parallelism is
            # enabled, because the KV cache managers might place the same
            # context on different devices/replicas.
            raise ValueError(
                "Having multiple KV managers (e.g. when using"
                " speculative decoding) is not supported when data "
                "parallelism is enabled."
            )
        batches = [{} for _ in range(data_parallel_degree)]
        for context in context_batch:
            req_id = context.request_id
            # Use whatever replica the main models KVCache recommends.
            replica_idx = kv_managers[0].get_or_recommend_replica(context)
            # Claim the slot for all kv_managers (eg: main + draft model)
            for kv_manager in self.kv_managers:
                kv_manager.claim(req_id, replica_idx=replica_idx)
            batches[replica_idx][req_id] = context
            batch_to_replica_idx[req_id] = replica_idx

        num_steps = self.pipeline_config.max_num_steps
        inputs = TextGenerationInputs(
            batches=batches,
            num_steps=num_steps,
        )

        # Generate outputs until all requests are done.

        done = 0

        try:
            while done < len(context_batch):
                for replica_batch in batches:
                    for ctx in replica_batch.values():
                        for kv_manager in self.kv_managers:
                            kv_manager.alloc(ctx, num_steps=num_steps)
                step_outputs = self.execute(inputs)
                outputs = []
                for request_id, output in step_outputs.items():
                    outputs.append(output)
                    if output.is_done:
                        done += 1
                        # Remove the request from the batch passed to the next
                        # call to execute.
                        replica_idx = batch_to_replica_idx[request_id]
                        batches[replica_idx].pop(request_id)

                        self.release(request_id)
                yield outputs

                # Yield to the event loop.  If at no other point (e.g.
                # tokenizer.decode which we await earlier does not yield to the
                # event loop), it will be at this point that we'll receive a
                # CancelledError if our future was canceled (e.g., we received a
                # SIGINT).
                await asyncio.sleep(0)
            assert all(len(batch) == 0 for batch in batches)
        finally:
            # Release remaining requests if the generation was interrupted.
            for batch in batches:
                for request_id in batch:
                    self.release(request_id)
