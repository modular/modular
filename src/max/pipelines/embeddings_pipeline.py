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

"""Pipeline for running text embeddings."""

from __future__ import annotations

from typing import Any, Type, TypeVar

from max.engine import InferenceSession
from max.profiler import Tracer, traced

from .config import PipelineConfig
from .context import InputContext
from .interfaces import EmbeddingsGenerator
from .pipeline import PipelineModel
from .response import EmbeddingsResponse

T = TypeVar("T", bound=InputContext)


class EmbeddingsPipeline(EmbeddingsGenerator[T]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: Type[PipelineModel],
        **unused_kwargs,
    ) -> None:
        self._pipeline_config = pipeline_config
        # Initialize Session.
        session = InferenceSession(devices=self._pipeline_config.devices)

        # Load model.
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config, session=session
        )

    @traced
    def encode(self, batch: dict[str, T]) -> dict[str, EmbeddingsResponse]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """

        tracer: Tracer = Tracer()
        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())

        tracer.next("prepare_initial_token_inputs")
        # Prepare inputs for the first token in multistep execution.
        model_inputs = self._pipeline_model.prepare_initial_token_inputs(
            context_batch
        )

        tracer.next("execute")
        model_outputs = self._pipeline_model.execute(
            model_inputs=model_inputs, kv_cache_inputs=None
        )
        assert model_outputs.logits
        # Do the copy to host for each token generated.
        tracer.next("logits.to(CPU())")
        batch_embeddings = model_outputs.logits.to_numpy()

        # Prepare the response.
        res: dict[str, Any] = {}
        tracer.push("prepare_response")
        for batch_index, request_id in enumerate(batch.keys()):
            request_embeddings = batch_embeddings[batch_index]
            if not self._pipeline_config.pool_embeddings:
                # Remove padded tokens from embeddings
                request_embeddings = request_embeddings[
                    : context_batch[batch_index].active_length, :
                ]
            res[request_id] = EmbeddingsResponse(request_embeddings)
        return res
