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

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental import functional as F
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)

from .batch_processor import Olmo3BatchProcessor
from .model_config import Olmo3Config
from .olmo3 import Olmo3

logger = logging.getLogger("max.pipelines")


@dataclass
class Olmo3Inputs(ModelInputs):
    """A class representing inputs for the Olmo3 model.

    This class encapsulates the input tensors required for the Olmo3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input sequence.
    """

    return_n_logits: Buffer
    """Number of logits to return."""


class Olmo3Model(PipelineModelWithKVCache[TextContext]):
    """An Olmo3 pipeline model for text generation.

    This class integrates the Olmo3 architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model_config_cls: ClassVar[type[Any]] = Olmo3Config
    batch_processor_cls: ClassVar[type[Olmo3BatchProcessor]] = (
        Olmo3BatchProcessor
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            return_logits: The number of top logits to return from the model
                execution.
        """
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model = self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Loads the compiled Olmo3 model into the MAX Engine session.

        Args:
            session: The MAX Engine inference session.

        Returns:
            The loaded MAX Engine model object.
        """

        with CompilationTimer("model") as timer:
            huggingface_config = self.huggingface_config
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }
            model_config = Olmo3Config.initialize(self.pipeline_config)
            model_config.finalize(
                huggingface_config=huggingface_config,
                state_dict=state_dict,
                return_logits=self.return_logits,
            )
            with F.lazy():
                nn_model = Olmo3(model_config, self.kv_params)
                nn_model.to(self.devices[0])

            assert self.batch_processor is not None
            compile_input_types = self.batch_processor.get_symbolic_inputs(
                kv_params=self.kv_params,
                device_refs=self.device_refs,
            )

            timer.mark_build_complete()
            compiled_model = nn_model.compile(
                *compile_input_types,
                weights=state_dict,
            )

        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Olmo3 model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        model_inputs = cast(Olmo3Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        if isinstance(model_inputs.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(
                model_inputs.input_row_offsets
            ).to(self.devices[0])
        else:
            input_row_offsets = model_inputs.input_row_offsets

        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            input_row_offsets,
            *curr_kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
            )
