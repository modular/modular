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
"""Implements the DeepseekV3 nn.model (ModuleV3)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from ..deepseekV2_modulev3.model import DeepseekV2Inputs, DeepseekV2Model
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


# DeepseekV3 reuses the same input layout as DeepseekV2 (tokens,
# input_row_offsets, return_n_logits, kv_cache_inputs).
DeepseekV3Inputs = DeepseekV2Inputs


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model (ModuleV3, single-GPU)."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return DeepseekV3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def load_model(self) -> Callable[..., Any]:
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV3."
            )

        huggingface_config = self.huggingface_config
        raw_state_dict = {
            key: value.data() for key, value in self.weights.items()
        }

        # Detect block-scaled FP8 quant config from the HF state dict
        # (uses the `weight_scale` substring match in the parser).
        dtype = self.dtype
        quant_config = None
        if dtype == DType.float8_e4m3fn:
            quant_config = parse_quant_config(
                huggingface_config, raw_state_dict, dtype
            )

        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = raw_state_dict

        model_config = DeepseekV3Config.initialize(self.pipeline_config)
        model_config.max_batch_context_length = (
            self.pipeline_config.runtime.max_batch_total_tokens
            or model_config.max_batch_context_length
        )
        model_config.quant_config = quant_config

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # When the weights are FP8, build the module with a bf16 default so
        # the non-quantized parameters (norms, biases, embeddings) match the
        # checkpoint's bf16 storage.
        module_default_dtype = (
            DType.bfloat16 if quant_config is not None else model_config.dtype
        )
        with F.lazy(), default_dtype(module_default_dtype):
            nn_model = DeepseekV3(model_config, self.kv_params)
            nn_model.to(self.devices[0])

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        return nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *flattened_kv_types,
            weights=state_dict,
        )
