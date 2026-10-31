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
"""The Mistral decoder the architecture registers for the ``DECODER`` role."""

from __future__ import annotations

from max import tree
from max.graph import TensorValue
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from .distributed_mistral import DistributedMistral
from .mistral import Mistral
from .model_config import MistralConfig

# The checkpoint names Mistral's weights from its root, so the decoder that
# holds one of these adds no prefix.


class _Mistral(Mistral):
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class _DistributedMistral(DistributedMistral):
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class MistralDecoder(Module):
    """Mistral, registered for the ``DECODER`` role.

    It runs on one device, or tensor-parallel across several.
    """

    graph_name = "mistral"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    # The checkpoint may hold vision tower weights (MODELS-551).
    strict_state_dict_loading = False

    def __init__(self, config: MistralConfig) -> None:
        """Builds the model for the config's devices."""
        super().__init__()
        self.kv_params = config.kv_params
        self.model: _Mistral | _DistributedMistral = (
            _DistributedMistral(config)
            if len(config.devices) > 1
            else _Mistral(config)
        )

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model on the batch and returns its logits."""
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(iter(tree.leaves(inputs.kv))),
            leaf=KVCacheInputsPerDevice,
        )
        if isinstance(self.model, _DistributedMistral):
            assert inputs.signals is not None
            return decoder_outputs(
                self.model(
                    inputs.tokens,
                    list(inputs.signals),
                    kv_collections,
                    inputs.return_n_logits,
                    inputs.input_row_offsets,
                )
            )
        return decoder_outputs(
            self.model(
                inputs.tokens,
                kv_collections[0],
                inputs.return_n_logits,
                inputs.input_row_offsets,
            )
        )
