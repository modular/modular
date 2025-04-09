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

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from max.driver import load_devices
from max.engine import InferenceSession
from max.graph.weights import load_weights
from max.nn import ReturnLogits
from max.profiler import Tracer, traced

if TYPE_CHECKING:
    from .config import PipelineConfig
from .core import EmbeddingsGenerator, EmbeddingsResponse, InputContext
from .hf_utils import download_weight_files
from .pipeline import PipelineModel

logger = logging.getLogger("max.pipelines")
T = TypeVar("T", bound=InputContext)


class EmbeddingsPipeline(EmbeddingsGenerator[T]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel],
        **unused_kwargs,
    ) -> None:
        self._pipeline_config = pipeline_config
        # Initialize Session.
        devices = load_devices(self._pipeline_config.model_config.device_specs)
        session = InferenceSession(devices=devices)

        if not self._pipeline_config.model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Download weight files if not existent
        weight_model_id = (
            self._pipeline_config.model_config._weights_repo_id
            if self._pipeline_config.model_config._weights_repo_id
            else self._pipeline_config.model_config.model_path
        )

        # Download weight files.
        weight_paths = download_weight_files(
            huggingface_model_id=weight_model_id,
            filenames=[
                str(x) for x in self._pipeline_config.model_config.weight_path
            ],
            revision=self._pipeline_config.model_config.huggingface_revision,
            force_download=self._pipeline_config.model_config.force_download,
        )

        # Load weights
        weights = load_weights(weight_paths)
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self._pipeline_config.model_config.huggingface_config,
            encoding=self._pipeline_config.model_config.quantization_encoding,
            devices=devices,
            kv_cache_config=self._pipeline_config.model_config.kv_cache_config,
            weights=weights,
            adapter=None,
            return_logits=ReturnLogits.ALL,
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
            context_batch=context_batch,
            kv_cache_inputs=None,
        )

        tracer.next("execute")
        model_outputs = self._pipeline_model.execute(model_inputs)

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
