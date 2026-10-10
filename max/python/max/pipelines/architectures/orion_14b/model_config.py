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

"""Config for Orion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.graph.weights import WeightData
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig


@dataclass(kw_only=True)
class OrionConfig(Llama3Config):
    """Model configuration for Orion graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float16",
        "float32",
    }

    layer_norm_eps: float = 1e-5
    """Epsilon for Orion's learned LayerNorm.

    Orion normalizes with ``nn.LayerNorm``, but its Hugging Face config stores
    the epsilon under ``rms_norm_eps`` -- a leftover from the Llama code Orion
    was forked from. ``Llama3Config.finalize`` leaves ``rms_norm_eps`` unset
    unless ``norm_method == "rms_norm"``, so the value is carried here instead.
    """

    interleaved_rope_weights: bool = False
    """Whether Orion's rotary weights use the GGUF interleaved layout.

    Always ``False``. MAX infers this from the weights format and assumes any
    GGUF checkpoint stores rotary weights permuted the way llama.cpp's
    converter emits them, but Orion's GGUF declares
    ``tensor_data_layout: "Meta AI original pth"`` and keeps the unpermuted
    layout its safetensors use. Without this override a GGUF load rotates
    Q/K incorrectly and degrades output without any error.
    """

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )
        self.layer_norm_eps = float(huggingface_config.rms_norm_eps)
        self.interleaved_rope_weights = False
