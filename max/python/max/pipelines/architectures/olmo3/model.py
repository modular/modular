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
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
from max import tree
from max.driver import Buffer
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
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


class Olmo3Model(
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """An Olmo3 pipeline model for text generation.

    This class integrates the Olmo3 architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model_config_cls: ClassVar[type[Any]] = Olmo3Config
    batch_processor_cls: ClassVar[type[Olmo3BatchProcessor]] = (
        Olmo3BatchProcessor
    )

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = Olmo3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Olmo3(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model

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
            *tree.leaves(curr_kv_cache_inputs),
        )
        return self._to_model_outputs(model_outputs)
