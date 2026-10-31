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
"""The Llama 4 decoder registered for the ``DECODER`` role."""

from __future__ import annotations

from max import tree
from max.graph import TensorValue
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from .llama4 import Llama4
from .model_config import Llama4Config


class _Llama4(Llama4):
    # The checkpoint names Llama 4's weights from its root, so the decoder
    # that holds it adds no prefix.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Llama4Decoder(Module):
    """Text-only Llama 4 on one device, registered for the ``DECODER`` role."""

    graph_name = "llama4"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    override_quantization_encoding = True
    strict_state_dict_loading = False

    def __init__(self, config: Llama4Config) -> None:
        super().__init__()
        self.model = _Llama4(config)
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
