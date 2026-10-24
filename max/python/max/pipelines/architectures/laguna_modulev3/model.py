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
"""Laguna ModuleV3 pipeline model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.laguna.model import (
    LagunaModel as LagunaGraphModel,
)
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModuleV3PipelineModelWithKVCache,
)
from typing_extensions import override

from .batch_processor import LagunaModuleV3BatchProcessor
from .laguna import Laguna
from .weight_adapters import to_graph_names

logger = logging.getLogger("max.pipelines")

_UNQUANTIZED_DTYPES = (DType.bfloat16, DType.float16, DType.float32)


def norm_dtype(state_dict: Mapping[str, WeightData]) -> DType:
    """Returns the one storage dtype every RMSNorm gamma shares.

    Args:
        state_dict: The checkpoint weights.

    Returns:
        The gammas' dtype, or bfloat16 when the checkpoint has none.

    Raises:
        ValueError: If the gammas disagree, which the single-dtype norm
            modules cannot represent.
    """
    dtypes = {
        data.dtype
        for name, data in state_dict.items()
        if name.endswith("norm.weight")
    }
    if len(dtypes) > 1:
        raise ValueError(
            "Laguna RMSNorm gammas must share one storage dtype, but the "
            f"checkpoint mixes {sorted(str(d) for d in dtypes)}."
        )
    return dtypes.pop() if dtypes else DType.bfloat16


class LagunaModel(ModuleV3PipelineModelWithKVCache[TextContext]):
    """Laguna pipeline model for text generation on ModuleV3.

    Single-GPU. Per-layer precision is read off the checkpoint by the graph
    architecture's resolver, so the two architectures agree on which layers
    are NVFP4.
    """

    model_config_cls: ClassVar[type[Any]] = LagunaConfig
    batch_processor_cls: ClassVar[type[LagunaModuleV3BatchProcessor]] = (
        LagunaModuleV3BatchProcessor
    )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> LagunaConfig:
        if len(self.devices) != 1:
            raise NotImplementedError(
                "Laguna on ModuleV3 runs on a single GPU; got "
                f"{len(self.devices)} devices."
            )
        if self.return_logits != ReturnLogits.LAST_TOKEN:
            raise NotImplementedError(
                "Laguna on ModuleV3 returns last-token logits only; got "
                f"{self.return_logits}."
            )
        model_config = LagunaConfig.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        # The config's detection is written against the graph architecture's
        # weight names, so it reads a view in those names.
        graph_state_dict = to_graph_names(state_dict)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=graph_state_dict,
            return_logits=self.return_logits,
        )
        # Dtype detection first: the quant resolution reads the sniffed
        # attention dtype.
        LagunaGraphModel._detect_state_dict_dtypes(
            model_config, graph_state_dict
        )
        LagunaGraphModel._resolve_nvfp4_quant_config(
            self.huggingface_config, model_config, graph_state_dict
        )
        if model_config.data_parallel_degree > 1:
            raise NotImplementedError(
                "Laguna on ModuleV3 does not support data parallelism."
            )
        attn_dtype = model_config.attn_dtype or model_config.unquantized_dtype
        if attn_dtype not in _UNQUANTIZED_DTYPES:
            raise NotImplementedError(
                "Laguna on ModuleV3 needs unquantized attention weights; the "
                f"checkpoint stores q_proj as {attn_dtype}."
            )
        self._norm_dtype = norm_dtype(graph_state_dict)
        return model_config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        # On an NVFP4 checkpoint ``model_config.dtype`` is the packed storage
        # dtype; the unquantized modules build in the checkpoint's own dtype.
        del state_dict
        return model_config.unquantized_dtype

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Laguna(model_config, self.kv_params, self._norm_dtype)
        nn_model.to(self.devices[0])
        return nn_model
