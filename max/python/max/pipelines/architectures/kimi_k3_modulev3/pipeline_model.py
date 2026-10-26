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
"""The Kimi K3 (ModuleV3) pipeline model.

``weight_adapters.convert_kimi_k3_state_dict`` already renames the
checkpoint to this module tree's names, so ``_prepare_state_dict`` is the
base class's no-op default.

No ``return_hidden_states`` support (``kimi_k3_output_head`` has no
per-layer capture for it).

EP and MXFP4 are mutually exclusive, matching ``moonshotai/Kimi-K3``
(MXFP4) vs ``moonshotai/Kimi-K3-ep`` (EP, bf16):
``expert_parallel_latent_moe`` doesn't support MXFP4 experts yet.
"""

from __future__ import annotations

from typing import Any, ClassVar

from max import tree
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPConfig, calculate_ep_max_tokens_per_rank
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModuleV3PipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import KimiK3ModuleV3BatchProcessor
from .layers import ep
from .model import (
    KimiK3Model,
    expert_parallel_kimi_k3_model,
    tensor_parallel_kimi_k3_model,
)
from .model_config import KimiK3Config

# `counters` is a mutable exchange-synchronization buffer; the other six
# leaves are read-only pointer tables.
_EP_BUFFER_TYPE_KINDS = (
    BufferType,
    TensorType,
    TensorType,
    TensorType,
    TensorType,
    TensorType,
    TensorType,
)


class KimiK3PipelineModel(ModuleV3PipelineModelWithKVCache[TextContext]):
    """Kimi K3's text tower, servable through the registry."""

    model_config_cls: ClassVar[type[Any]] = KimiK3Config
    batch_processor_cls: ClassVar[type[KimiK3ModuleV3BatchProcessor]] = (
        KimiK3ModuleV3BatchProcessor
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
        # `super().__init__` calls `_init_distributed_runtime`, so the
        # session must already be set here.
        self.session = session
        self._ep_leaf_buffers: tuple[Any, ...] = ()
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

    def _create_model_config(self, state_dict: dict[str, Any]) -> KimiK3Config:
        config = self.arch_config_as(KimiK3Config)
        config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or config.max_batch_context_length
        )
        if self.dtype == DType.uint8:
            config.quant_config = QuantConfig.mxfp4(
                mlp_quantized_layers=range(
                    config.model.first_k_dense_replace,
                    config.model.num_hidden_layers,
                )
            )
        n_devices = len(self.devices)
        axis_name = "ep" if config.ep_size > 1 else "tp"
        config.mesh = DeviceMesh(
            tuple(self.devices), (n_devices,), (axis_name,)
        )
        return config

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict
        # Everything but the routed experts is bf16 under MXFP4.
        if model_config.quant_config is not None:
            return DType.bfloat16
        return model_config.dtype

    def _init_distributed_runtime(self, model_config: KimiK3Config) -> None:
        super()._init_distributed_runtime(model_config)
        if model_config.ep_size <= 1 or model_config.quant_config is not None:
            return
        n_devices = len(self.devices)
        ep_size = model_config.ep_size
        if ep_size % n_devices != 0:
            raise ValueError(
                f"Kimi K3: ep_size={ep_size} must be divisible by the"
                f" number of GPUs on this node ({n_devices}); for"
                f" single-node set ep_size={n_devices}."
            )
        assert model_config.mesh is not None
        ep_config = EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.model.routed_expert_hidden_size,
            top_k=model_config.model.num_experts_per_token,
            n_experts=model_config.model.num_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=1,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=ep_size // n_devices,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )
        model_config.ep_config = ep_config
        ep_buffers = ep.allocate_ep_buffers(
            ep_config, model_config.mesh, self.session
        )

        # Not a checkpoint weight, so `forward` reads it off the tail of
        # `kv_inputs` (see `KimiK3Model.__init__`) instead of a stored
        # attribute, which `Module.stage` would declare as one. `counters`
        # is sharded across the mesh, one shard per device; the other six
        # leaves are each a single CPU pointer table.
        extra_types: list[TensorType | BufferType] = []
        extra_buffers: list[Any] = []
        for leaf, type_kind in zip(
            tree.leaves(ep_buffers), _EP_BUFFER_TYPE_KINDS, strict=True
        ):
            for shard in leaf.buffers:
                extra_types.append(
                    type_kind(
                        shard.dtype,
                        shard.shape,
                        device=DeviceRef.from_device(shard.device),
                    )
                )
                extra_buffers.append(shard)
        self._modulev3_extra_input_types.extend(extra_types)
        self._ep_leaf_buffers = tuple(extra_buffers)

    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        # `ModuleV3PipelineModelWithKVCache` never calls `_wire_batch_processor`
        # (that hook only fires for `GraphPipelineModelWithKVCache`), so bind
        # the EP buffers here instead, right after compile.
        models = super().load_graphs(session)
        if self._ep_leaf_buffers:
            batch_processor = self.batch_processor
            assert isinstance(batch_processor, KimiK3ModuleV3BatchProcessor)
            batch_processor.bind_ep_buffers(self._ep_leaf_buffers)
        return models

    def _instantiate_module(self, model_config: KimiK3Config) -> KimiK3Model:
        assert model_config.mesh is not None
        mesh = model_config.mesh
        n_devices = len(self.devices)
        with default_device(mesh):
            module = KimiK3Model(
                model_config.model,
                kv_params=model_config.kv_params,
                return_logits=self.return_logits,
                mla_buffer_size=model_config.max_batch_context_length,
                quant_config=model_config.quant_config,
                ep_config=model_config.ep_config,
            )
            if model_config.ep_config is not None:
                expert_parallel_kimi_k3_model(module, mesh)
            elif n_devices > 1:
                tensor_parallel_kimi_k3_model(module, mesh)
            return module
