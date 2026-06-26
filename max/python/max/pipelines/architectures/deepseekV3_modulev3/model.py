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
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from ..deepseekV2_modulev3.model import DeepseekV2Inputs, DeepseekV2Model
from .batch_processor import DeepseekV3ModuleV3BatchProcessor
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


# DeepseekV3 reuses the same input layout as DeepseekV2 (tokens,
# input_row_offsets, return_n_logits, kv_cache_inputs).
DeepseekV3Inputs = DeepseekV2Inputs


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model (ModuleV3), single- or multi-GPU (TP attention, EP)."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config
    batch_processor_cls: ClassVar[type[DeepseekV3ModuleV3BatchProcessor]] = (
        DeepseekV3ModuleV3BatchProcessor
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
        # Capture the session so load_model() can initialize EP communication,
        # and default the EP buffers so execute() works without EP.
        self.session = session
        self._ep_model_inputs: list[Buffer] = []
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

    def _build_ep_config(
        self, ep_size: int, n_devices: int, model_config: DeepseekV3Config
    ) -> EPConfig:
        """Build the expert-parallel config from the pipeline ep_size."""
        if ep_size % n_devices != 0:
            raise ValueError(
                f"ep_size={ep_size} must be divisible by the number of GPUs on"
                f" this node ({n_devices}); for single-node set"
                f" ep_size={n_devices}."
            )
        ep_max_tokens_per_rank = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )
        if model_config.n_shared_experts == 1:
            # Only enable shared expert fusion if the shared expert is of
            # the same shape as routed experts.
            fused_shared_expert = True
        return EPConfig(
            # Dispatch tokens in bf16 regardless of weight dtype; FP8 experts
            # quantize the activations locally in the grouped matmul. This
            # avoids FP8-on-the-wire dispatch (and its dispatch_quant_config),
            # so ``dispatch_quant_config`` stays ``None`` (required by
            # ``call_ep_init`` for a bf16 dispatch dtype).
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.hidden_size,
            top_k=model_config.num_experts_per_tok,
            n_experts=model_config.n_routed_experts,
            max_tokens_per_rank=ep_max_tokens_per_rank,
            n_gpus_per_node=n_devices,
            n_nodes=ep_size // n_devices,
            fused_shared_expert=fused_shared_expert,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )

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

        if model_config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            model_config.correction_bias_dtype = state_dict[
                correction_bias_key
            ].dtype

        # Tensor-parallel device mesh across all devices (single-device mesh
        # for single-GPU runs). Drives weight placement and the collectives
        # inserted by the sharding propagation.
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        model_config.mesh = mesh

        # Expert parallelism: ep_size > 1 distributes routed experts across the
        # devices via the NVSHMEM EPBatchManager. The communication buffers are
        # allocated once here and threaded through the graph as extra inputs.
        ep_size = self.pipeline_config.runtime.ep_size
        ep_batch_manager: EPBatchManager | None = None
        ep_input_types: list[Any] = []
        self._ep_model_inputs = []
        if ep_size > 1:
            ep_config = self._build_ep_config(ep_size, n_devices, model_config)
            model_config.ep_config = ep_config
            ep_batch_manager = EPBatchManager(ep_config)
            ep_input_types = ep_batch_manager.input_types()
            if not is_virtual_device_mode():
                ep_comm_initializer = EPCommInitializer(ep_config)
                ep_comm_initializer.ep_init(self.session)
                ep_config.node_id = ep_comm_initializer.config.node_id
                self._ep_model_inputs = ep_comm_initializer.model_inputs()

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # When the weights are FP8, build the module with a bf16 default so
        # the non-quantized parameters (norms, biases, embeddings) match the
        # checkpoint's bf16 storage.
        module_default_dtype = (
            DType.bfloat16 if quant_config is not None else model_config.dtype
        )
        with F.lazy(), default_dtype(module_default_dtype):
            nn_model = DeepseekV3(
                model_config, self.kv_params, ep_batch_manager
            )
            nn_model.to(mesh)

        assert isinstance(
            self.batch_processor, DeepseekV3ModuleV3BatchProcessor
        )
        compile_input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params,
            device_refs=[device_ref],
            extra_input_types=ep_input_types,
        )

        return nn_model.compile(
            *compile_input_types,
            weights=state_dict,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        assert isinstance(model_inputs, DeepseekV3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *curr_kv_cache_inputs.flatten(),
            *self._ep_model_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        return ModelOutputs(
            logits=cast(Buffer, model_outputs[0].driver_tensor),
            next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
        )
