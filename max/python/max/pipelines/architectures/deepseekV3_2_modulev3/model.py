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
"""Implements the DeepseekV3.2 nn.model (ModuleV3)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from max.dtype import DType
from max.experimental.tensor import default_device
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3_modulev3.model import DeepseekV3Model
from .deepseekV3_2 import DeepseekV3_2
from .model_config import DeepseekV3_2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3_2Model(DeepseekV3Model):
    """A DeepseekV3.2 model (ModuleV3).

    Inherits the V3 weight loading, quant-config derivation, mesh setup and
    expert-parallel runtime; only the config class and the module differ.
    """

    model_config_cls: ClassVar[type[Any]] = DeepseekV3_2Config

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return DeepseekV3_2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        assert model_config.mesh is not None
        with default_device(model_config.mesh):
            return DeepseekV3_2(
                model_config, self.kv_params, self._ep_batch_manager
            )
