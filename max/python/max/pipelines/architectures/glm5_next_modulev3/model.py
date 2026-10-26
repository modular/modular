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
"""The GLM-5.3-Flash (``glm5_next``) pipeline model on ModuleV3."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from max.dtype import DType
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3_2_modulev3.model import DeepseekV3_2Model
from .glm5_next import Glm5Next
from .model_config import Glm5NextConfig

logger = logging.getLogger("max.pipelines")

__all__ = ["Glm5NextModel"]


class Glm5NextModel(DeepseekV3_2Model):
    """GLM-5.3-Flash pipeline model (ModuleV3).

    Inherits DeepSeek-V3.2's weight loading, mesh and expert-parallel runtime.
    The config is GLM-5.3-Flash's: its cache tree adds the KDA state, and its
    precision map is read off the checkpoint rather than assumed uniform.
    """

    model_config_cls: ClassVar[type[Any]] = Glm5NextConfig

    # The decoder shards the MoE rows across the TP ranks before the expert
    # dispatch (`Glm5NextDecoderLayer.forward`), so exporters pick the P2P
    # EP backend.
    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return Glm5NextConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        config = Glm5NextConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or config.max_batch_context_length
        )
        config.quant_config = config.resolve_quant_scheme(state_dict)
        for name, weight in state_dict.items():
            if name.endswith("e_score_correction_bias"):
                config.correction_bias_dtype = weight.dtype
                break
        n_devices = len(self.devices)
        config.mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))

        logger.info(
            "GLM-5.3-Flash pools: %d KDA layers holding %.1f MiB per sequence, "
            "context-independent; %d sparse-MLA layers.",
            len(config.kda_layers),
            config.per_request_state_bytes() / 1024**2,
            len(config.sparse_attention_layers),
        )
        return config

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        # The graph returns the last token's logits only; a caller asking for
        # more would otherwise get those silently.
        if (
            self.return_logits is not ReturnLogits.LAST_TOKEN
            or self.return_hidden_states is not ReturnHiddenStates.NONE
        ):
            raise NotImplementedError(
                f"return_logits={self.return_logits} with "
                f"return_hidden_states={self.return_hidden_states} is not "
                "implemented for GLM-5.3-Flash on ModuleV3; only last-token "
                "logits are."
            )
        assert model_config.mesh is not None
        with default_device(model_config.mesh):
            return Glm5Next(
                model_config, self.kv_params, self._ep_batch_manager
            )
