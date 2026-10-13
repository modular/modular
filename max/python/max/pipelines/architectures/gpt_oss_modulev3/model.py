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

from .batch_processor import GptOssModuleV3BatchProcessor
from .gpt_oss import GptOss
from .inputs import GptOssInputs
from .model_config import GptOssConfig

logger = logging.getLogger("max.pipelines")


class GptOssModel(ModuleV3PipelineModelWithKVCache[TextContext]):
    """A GPT OSS pipeline model for text generation.

    This class integrates the GPT OSS architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model_config_cls: ClassVar[type[Any]] = GptOssConfig
    batch_processor_cls: ClassVar[type[GptOssModuleV3BatchProcessor]] = (
        GptOssModuleV3BatchProcessor
    )

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = GptOssConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = GptOss(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the GPT OSS model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        model_inputs = cast(GptOssInputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        # For backward compatibility, distribute the single tensor to all devices
        if isinstance(model_inputs.input_row_offsets, np.ndarray):
            # Convert numpy array to tensor first
            tensor = Buffer.from_numpy(model_inputs.input_row_offsets)
            input_row_offsets = tensor.to(self.devices[0])
        else:
            # Already a tensor
            input_row_offsets = model_inputs.input_row_offsets

        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            input_row_offsets,
            *tree.leaves(curr_kv_cache_inputs),
        )
        return self._to_model_outputs(model_outputs)
