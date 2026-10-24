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
"""The Hy4 pipeline model (ModuleV3)."""

from __future__ import annotations

from typing import Any, ClassVar

from max.dtype import DType
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache, PipelineConfig
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from .batch_processor import HYV4BatchProcessor
from .hy_v4 import HYV4
from .model_config import HYV4Config


def check_parallelism(pipeline_config: PipelineConfig, n_devices: int) -> None:
    """Raises unless several devices run tensor-parallel attention.

    The routed experts split over the devices either way: each device runs
    its own on every token and the outputs are summed, with no expert
    dispatch, so ``ep_size`` may be 1 or the device count.
    """
    dp_degree = pipeline_config.model.data_parallel_degree
    ep_size = pipeline_config.runtime.ep_size
    if n_devices > 1 and (dp_degree != 1 or ep_size not in (1, n_devices)):
        raise ValueError(
            f"Hy4 on {n_devices} devices runs tensor-parallel attention with"
            " its experts split over the devices: it needs"
            f" data_parallel_degree 1 and ep_size 1 or {n_devices}, got"
            f" {dp_degree} and {ep_size}"
        )


class HYV4Model(
    LogProbabilitiesMixin, ModuleV3PipelineModelWithKVCache[TextContext]
):
    """Hy4 on one GPU, or on several with tensor-parallel attention and
    expert-parallel MoE."""

    model_config_cls: ClassVar[type[Any]] = HYV4Config
    batch_processor_cls: ClassVar[type[HYV4BatchProcessor]] = HYV4BatchProcessor

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> HYV4Config:
        check_parallelism(self.pipeline_config, len(self.devices))
        config = HYV4Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or config.max_batch_context_length
        )
        config.quant_config = parse_quant_config(
            self.huggingface_config, state_dict, self.dtype
        )
        return config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        # The FP8 checkpoint stores every unquantized tensor in BF16 or fp32;
        # fp32 ones (iHC, sinks, router bias) are declared explicitly.
        del state_dict, model_config
        return DType.bfloat16

    @override
    def _instantiate_module(self, model_config: HYV4Config) -> HYV4:
        if len(self.devices) == 1:
            return HYV4(model_config).to(self.devices[0])
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), (TP,))
        with default_device(mesh):
            model = HYV4(model_config)
        model.language_model.shard()
        return model
