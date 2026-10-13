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
"""GLM-5.x MTP with a ModuleV3 target and draft, compiled as one module."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cached_property
from typing import Any, ClassVar

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.tensor import default_device, default_dtype
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.kv_cache import MultiKVCacheParams
from max.profiler import traced
from typing_extensions import override

from ..deepseekV3_2_modulev3.deepseekV3_2 import DeepseekV3_2
from ..deepseekV3_2_modulev3.model_config import DeepseekV3_2Config
from ..deepseekV3_2_modulev3.nextn import DeepseekV3_2NextN
from ..deepseekV3_modulev3.model import build_ep_config, create_model_config
from ..glm5_1_modulev3.model_config import Glm5_1Config
from ..unified_mtp_glm5_2.model import UnifiedMTPGlm5_2Model, _subtree_quantized
from .batch_processor import UnifiedMTPGlm5_2ModuleV3BatchProcessor
from .unified_mtp_glm5_2 import UnifiedMTPGlm5_2


class UnifiedMTPGlm5_2ModuleV3Model(UnifiedMTPGlm5_2Model):
    """GLM-5.x with MTP on ModuleV3: target, draft and driver in one module.

    Weight loading, EP initialization and batching are the graph-API MTP
    model's. The graph is :meth:`~max.experimental.nn.Module.compile`'s, so its
    signal buffers trail every other input.
    """

    batch_processor_cls: ClassVar[
        type[UnifiedMTPGlm5_2ModuleV3BatchProcessor]
    ] = UnifiedMTPGlm5_2ModuleV3BatchProcessor

    _target_config: DeepseekV3_2Config
    _compiled: CompiledCallable[..., tuple[Any, ...]]

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        # A compiled graph declares signal buffers only when it spans more
        # than one device.
        if len(self.devices) <= 1:
            return []
        return Signals.allocate(self.devices)

    def _create_target_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV3_2Config:
        if self.pipeline_config.model.data_parallel_degree > 1:
            raise NotImplementedError(
                "MTP with a ModuleV3 GLM target requires "
                "data_parallel_degree=1."
            )
        n_devices = len(self.devices)
        ep_size = self.pipeline_config.runtime.ep_size
        if n_devices > 1 and ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        target_config = create_model_config(self, Glm5_1Config, state_dict)
        assert isinstance(target_config, DeepseekV3_2Config)
        if ep_size > 1:
            target_config.ep_config = build_ep_config(
                self, ep_size, n_devices, target_config
            )
        self._split_mtp_kv_params()
        self._target_config = target_config
        return target_config

    @traced
    @override
    def load_model(self, session: InferenceSession) -> Model:
        target_state_dict = self._load_state_dict()
        model_config = self._create_target_config(target_state_dict)
        self._init_distributed_runtime(session, model_config)

        module = self._build_module()
        self._compiled = module.compile(
            *module.input_types(),
            weights={
                **{f"target.{k}": v for k, v in target_state_dict.items()},
                **{f"draft.{k}": v for k, v in self._draft_state_dict.items()},
            },
        )
        self.state_dict = dict(self._compiled.weights)
        model = self._compiled.engine_model
        self._wire_batch_processor(model, model_config)
        return model

    def _build_module(self) -> UnifiedMTPGlm5_2:
        target_config = self._target_config
        mesh = target_config.mesh
        assert mesh is not None
        assert self.pipeline_config.speculative is not None
        assert isinstance(self.kv_params, MultiKVCacheParams)
        target_kv_params = self.kv_params.children["target"]
        assert isinstance(target_kv_params, MultiKVCacheParams)
        draft_config = self._draft_module_config()
        # The quantized checkpoint stores its norms, biases and embeddings
        # in bf16.
        module_dtype = (
            DType.bfloat16
            if target_config.quant_config is not None
            else target_config.dtype
        )
        with F.lazy(), default_dtype(module_dtype), default_device(mesh):
            # Built only now: ``_init_distributed_runtime`` sets the node id.
            target = DeepseekV3_2(
                target_config,
                target_kv_params,
                EPBatchManager(target_config.ep_config)
                if target_config.ep_config is not None
                else None,
            )
            qk_head_dim = (
                target_config.qk_rope_head_dim + target_config.qk_nope_head_dim
            )
            draft = DeepseekV3_2NextN(
                draft_config,
                attention_scale=target.language_model.rope.compute_scale(
                    math.sqrt(1.0 / qk_head_dim)
                ),
                ep_batch_manager=EPBatchManager(draft_config.ep_config)
                if draft_config.ep_config is not None
                else None,
            )
            return UnifiedMTPGlm5_2(
                target,
                draft,
                kv_params=self.kv_params,
                speculative_config=self.pipeline_config.speculative,
                enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            )

    def _draft_module_config(self) -> DeepseekV3_2Config:
        """The target's config, narrowed to the single MTP layer.

        The layer owns a single-layer cache, computes its own top-k, and may
        be quantized differently from the target's layers: NVFP4 checkpoints
        leave it in bf16.
        """
        target_config = self._target_config
        assert isinstance(self.kv_params, MultiKVCacheParams)
        draft_kv_params = self.kv_params.children["draft"]
        assert isinstance(draft_kv_params, MultiKVCacheParams)
        layer_idx = max(
            target_config.num_hidden_layers,
            target_config.first_k_dense_replace,
        )
        experts_quantized = _subtree_quantized(
            self._draft_state_dict, ".mlp.experts."
        )

        quant_config = target_config.quant_config
        if quant_config is not None:
            mlp_layers = set(quant_config.mlp_quantized_layers)
            attn_layers = set(quant_config.attn_quantized_layers)
            if experts_quantized:
                mlp_layers.add(layer_idx)
            if _subtree_quantized(self._draft_state_dict, ".self_attn."):
                attn_layers.add(layer_idx)
            quant_config = replace(
                quant_config,
                mlp_quantized_layers=mlp_layers,
                attn_quantized_layers=attn_layers,
            )

        ep_config = target_config.ep_config
        if ep_config is not None and not experts_quantized:
            ep_config = replace(
                ep_config,
                dispatch_dtype=DType.bfloat16,
                dispatch_quant_config=None,
                fused_shared_expert=target_config.n_shared_experts == 1,
            )

        return replace(
            target_config,
            kv_params=draft_kv_params,
            indexer_types=[],
            quant_config=quant_config,
            ep_config=ep_config,
        )
