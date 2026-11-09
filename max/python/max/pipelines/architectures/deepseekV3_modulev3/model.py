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
from typing import Any, ClassVar

from max.driver import Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.ep_config import EPConfig
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer
from max.nn.outputs import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.tensor import default_device
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV2_modulev3.model import DeepseekV2Model
from .batch_processor import DeepseekV3ModuleV3BatchProcessor
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


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
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        # Capture the session so _init_distributed_runtime() can initialize EP
        # communication, and default the EP buffers so execute() works without EP.
        self.session = session
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

    def _module_runtime(self) -> ModuleRuntime:
        """Returns the deployment the config is completed with."""
        runtime = self.pipeline_config.runtime
        n_devices = len(self.devices)
        dp_degree = self.pipeline_config.model.data_parallel_degree
        return ModuleRuntime(
            mesh=DeviceMesh(
                tuple(self.devices),
                (n_devices,),
                (DP if dp_degree > 1 else TP,),
            ),
            data_parallel_degree=dp_degree,
            max_batch_input_tokens=runtime.max_batch_input_tokens,
            max_batch_total_tokens=self.planned_max_batch_total_tokens,
            ep_size=runtime.ep_size,
            ep_use_allreduce=runtime.ep_use_allreduce,
        )

    def _build_ep_config(self, model_config: DeepseekV3Config) -> EPConfig:
        """Build the expert-parallel config from the pipeline ep_size."""
        if self.pipeline_config.runtime.ep_fuse_ffn_combine_send:
            logger.warning(
                "ep_fuse_ffn_combine_send is not implemented on ModuleV3; the"
                " MoE FFN output is materialized before the combine."
            )
        return model_config.expert_parallel_config(self._module_runtime())

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """The KV cache parameters this model was traced with."""
        return DeepseekV3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV3."
            )
        return super()._load_state_dict()

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = self.model_config_cls.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config = model_config.finalize(
            self.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            ),
        )
        return model_config.apply_runtime(self._module_runtime())

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict
        # When the weights are FP8, build the module with a bf16 default so
        # the non-quantized parameters (norms, biases, embeddings) match the
        # checkpoint's bf16 storage.
        if model_config.quant_config is not None:
            return DType.bfloat16
        return model_config.dtype

    @override
    def _init_distributed_runtime(self, model_config: Any) -> None:
        super()._init_distributed_runtime(model_config)
        self._ep_batch_manager = None

        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return

        # Expert parallelism: ep_size > 1 distributes routed experts across the
        # devices via the NVSHMEM EPBatchManager. The communication buffers are
        # allocated once here and threaded through the graph as extra inputs.
        if ep_size % len(self.devices) != 0:
            raise ValueError(
                f"ep_size={ep_size} must be divisible by the number of GPUs on"
                f" this node ({len(self.devices)}); for single-node set"
                f" ep_size={len(self.devices)}."
            )
        ep_config = self._build_ep_config(model_config)
        model_config.ep_config = ep_config
        self._ep_batch_manager = EPBatchManager(ep_config)
        self._modulev3_extra_input_types = self._ep_batch_manager.input_types()
        if not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(ep_config)
            self.ep_comm_initializer.ep_init(self.session)
            ep_config.node_id = self.ep_comm_initializer.config.node_id

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        assert model_config.mesh is not None
        with default_device(model_config.mesh):
            return DeepseekV3(
                model_config, self.kv_params, self._ep_batch_manager
            )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        return self._to_model_outputs(self.model(*model_inputs.buffers))
