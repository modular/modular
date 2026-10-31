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
"""The Llama 3 decoder an architecture registers for the ``DECODER`` role."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from max import tree
from max.graph import TensorValue
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from transformers import AutoConfig

from .llama3 import Llama3
from .model_config import Llama3Config


@dataclass(kw_only=True)
class Llama3DecoderConfig(Llama3Config):
    """The config an architecture registers with :class:`Llama3Decoder`.

    The framework finalizes it with the outputs the pipeline requires.
    """

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        return_logits: ReturnLogits | ModelOutputConfig,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Completes the config in place with checkpoint-dependent fields."""
        if isinstance(return_logits, ModelOutputConfig):
            return_hidden_states = return_logits.return_hidden_states
            return_logits = return_logits.return_logits
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=dict(state_dict),
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )


class _Llama3(Llama3):
    # The checkpoint names Llama 3's weights from its root, so the decoder
    # that holds it adds no prefix.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Llama3Decoder(Module):
    """Llama 3 on one device, registered for the ``DECODER`` role.

    TODO(kathywu): Register the tensor- and data-parallel graphs too, so
    Llama 3 itself and its multi-GPU derivatives can register this.
    """

    graph_name = "llama3"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    override_quantization_encoding = True
    # The checkpoint's `rope_freqs.weight` isn't read (MODELS-550).
    strict_state_dict_loading = False

    def __init__(self, config: Llama3Config) -> None:
        """Builds the model.

        Raises:
            NotImplementedError: If the config spans several devices.
        """
        super().__init__()
        if len(config.devices) != 1:
            raise NotImplementedError(
                "the registered Llama 3 decoder runs on one device"
            )
        self.model = _Llama3(config)
        self.kv_params = config.kv_params

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model on the batch and returns its logits."""
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(iter(tree.leaves(inputs.kv))),
            leaf=KVCacheInputsPerDevice,
        )
        return decoder_outputs(
            self.model(
                inputs.tokens,
                kv_collections[0],
                inputs.return_n_logits,
                inputs.input_row_offsets,
            )
        )
