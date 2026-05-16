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
"""Defines the MPNet V3 pipeline model.

Implementation is based on MPNetModel from the transformers library,
using the V3 eager API (max.experimental.nn).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.buffer_utils import cast_tensor_to
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    upper_bounded_default,
)
from transformers import AutoConfig

from .graph import MPNetModel
from .model_config import MPNetConfig

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 1


@dataclass
class MPNetInputs(ModelInputs):
    """Input tensors for the MPNet model."""

    next_tokens_batch: Buffer
    attention_mask: Buffer


class MPNetPipelineModel(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
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

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for MPNet, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MPNetInputs)
        model_outputs = self.model(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        result = model_outputs[0].driver_tensor
        assert isinstance(result, Buffer)
        return ModelOutputs(logits=result)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> MPNetInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        # Get tokens and seq_ids.
        tokens = [ctx.tokens.active for ctx in context_batch]

        # Pad tokens for the batch.
        pad_value = getattr(self.huggingface_config, "pad_token_id", 1)
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )

        # Compute attention mask.
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)

        return MPNetInputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(
                self.devices[0]
            ),
            attention_mask=Buffer.from_numpy(attention_mask).to(
                self.devices[0]
            ),
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> MPNetInputs:
        raise NotImplementedError(
            "MPNet does not support preparing next tokens inputs."
        )

    def load_model(self) -> Callable[..., list[Tensor]]:
        if self.adapter:
            state_dict = self.adapter(dict(self.weights.items()))
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        # Cast weights to match the model's configured dtype (e.g. float32
        # safetensor weights -> bfloat16 when default_encoding is bfloat16).
        # V3 compile() requires exact dtype matching unlike V2 load_state_dict.
        target_dtype = self.dtype
        cast_state_dict: dict[str, Any] = {}
        for k, v in state_dict.items():
            buf = Buffer.from_dlpack(v) if not isinstance(v, Buffer) else v
            if buf.dtype != target_dtype and buf.dtype.is_float():
                buf = cast_tensor_to(buf, target_dtype)
            cast_state_dict[k] = buf
        state_dict = cast_state_dict

        config = MPNetConfig.initialize(self.pipeline_config)

        with F.lazy(), default_dtype(target_dtype):
            nn_model = MPNetModel(config)
            nn_model.to(self.devices[0])

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        input_ids_type = TensorType(
            DType.int64, shape=["batch_size", "seq_len"], device=device_ref
        )
        attention_mask_type = TensorType(
            DType.float32,
            shape=["batch_size", "seq_len"],
            device=device_ref,
        )

        compiled_model = nn_model.compile(
            input_ids_type,
            attention_mask_type,
            weights=state_dict,
        )

        return compiled_model
