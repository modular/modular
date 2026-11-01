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
"""The OLMo 2 decoder the architecture registers for the ``DECODER`` role."""

from __future__ import annotations

from max import tree
from max.graph import TensorValue
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from .model_config import Olmo2Config
from .olmo2 import Olmo2


class _Olmo2(Olmo2):
    # The checkpoint names OLMo 2's weights from its root, so the decoder
    # that holds it adds no prefix.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Olmo2Decoder(Module):
    """OLMo 2 on one device, registered for the ``DECODER`` role."""

    graph_name = "olmo2"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    override_quantization_encoding = True

    def __init__(self, config: Olmo2Config) -> None:
        """Builds the model.

        Raises:
            NotImplementedError: If the config spans several devices.
        """
        super().__init__()
        if len(config.devices) != 1:
            raise NotImplementedError(
                "the registered OLMo 2 decoder runs on one device"
            )
        self.model = _Olmo2(config)
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
                return_n_logits=inputs.return_n_logits,
                input_row_offsets=inputs.input_row_offsets,
            )
        )
