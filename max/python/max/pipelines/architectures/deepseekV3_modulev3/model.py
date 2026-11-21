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

from typing import Any, ClassVar

from max import tree
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.ep import init as ep_init
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.lib import ModelInputs, ModelOutputs
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_runtime import (
    ExpertParallelInputs,
    ModuleRuntime,
)
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV2_modulev3.model import DeepseekV2Model
from .batch_processor import DeepseekV3ModuleV3BatchProcessor
from .model_config import DeepseekV3Config


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model (ModuleV3), single- or multi-GPU (TP attention, EP)."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config
    batch_processor_cls: ClassVar[type[DeepseekV3ModuleV3BatchProcessor]] = (
        DeepseekV3ModuleV3BatchProcessor
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
        if self.pipeline_config.runtime.ep_size > 1:
            # The decoder takes the exchange's buffers after its other
            # inputs, flat in the order of the registered decoder's ``ep``.
            ep, self.ep_comm_initializer, modules = (
                ExpertParallelInputs().initialize(
                    self._session,
                    model_config.expert_parallel_config(self._module_runtime()),
                    model_config.mesh,
                )
            )
            self.compiled_modules = {**self.compiled_modules, **modules}
            model_config.ep_config = ep
            self._modulev3_extra_input_types = tree.leaves(
                ep_init.payload_types(ep)
            )
        # Hybrid HiSparse (a partly host-resident MLA latent) appends its three
        # input groups after the EP buffers. Only the V3.2 config family
        # carries the field.
        if getattr(model_config, "hisparse", None) is not None:
            self._modulev3_extra_input_types.extend(
                model_config.hisparse_input_types()
            )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        return self._to_model_outputs(self.model(*model_inputs.buffers))
