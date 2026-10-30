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
"""Config for the Laguna ModuleV3 decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.laguna.model import (
    LagunaModel as LagunaGraphModel,
)
from max.pipelines.architectures.laguna.model_config import LagunaConfig
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from transformers import AutoConfig

from .weight_adapters import to_graph_names

_UNQUANTIZED_DTYPES = (DType.bfloat16, DType.float16, DType.float32)


def _gamma_dtype(state_dict: Mapping[str, WeightData]) -> DType:
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


@dataclass(kw_only=True)
class LagunaDecoderConfig(LagunaConfig):
    """The config of the registered :class:`~.laguna.Laguna` decoder.

    Single-GPU. Per-layer precision is read off the checkpoint by the graph
    architecture's resolver, so the two architectures agree on which layers
    are NVFP4.
    """

    norm_dtype: DType | None = None
    """The one storage dtype of every RMSNorm gamma, set by :meth:`finalize`."""

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits | ModelOutputConfig,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Completes the config in place with checkpoint-dependent fields.

        The framework finalizes a copy, so the registered config itself never
        changes.

        Raises:
            NotImplementedError: If the deployment needs more than one device,
                data parallelism or other than last-token logits, or the
                checkpoint's attention weights are quantized.
        """
        if isinstance(return_logits, ModelOutputConfig):
            return_hidden_states = return_logits.return_hidden_states
            return_logits = return_logits.return_logits
        if len(self.devices) != 1:
            raise NotImplementedError(
                "Laguna on ModuleV3 runs on a single GPU; got "
                f"{len(self.devices)} devices."
            )
        if return_logits != ReturnLogits.LAST_TOKEN:
            raise NotImplementedError(
                "Laguna on ModuleV3 returns last-token logits only; got "
                f"{return_logits}."
            )
        if self.data_parallel_degree > 1:
            raise NotImplementedError(
                "Laguna on ModuleV3 does not support data parallelism."
            )
        # The detection below is written against the graph architecture's
        # weight names, so it reads a view in those names.
        graph_state_dict = to_graph_names(state_dict)
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=graph_state_dict,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )
        # Dtype detection first: the quant resolution reads the sniffed
        # attention dtype.
        LagunaGraphModel._detect_state_dict_dtypes(self, graph_state_dict)
        LagunaGraphModel._resolve_nvfp4_quant_config(
            huggingface_config, self, graph_state_dict
        )
        attn_dtype = self.attn_dtype or self.unquantized_dtype
        if attn_dtype not in _UNQUANTIZED_DTYPES:
            raise NotImplementedError(
                "Laguna on ModuleV3 needs unquantized attention weights; the "
                f"checkpoint stores q_proj as {attn_dtype}."
            )
        self.norm_dtype = _gamma_dtype(graph_state_dict)
