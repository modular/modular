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
"""The Qwen3 decoder the dense architecture registers for ``DECODER``."""

from __future__ import annotations

from collections.abc import Mapping

from max import tree
from max.driver import DLPackArray
from max.graph import TensorValue
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.layer import Module
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from .model_config import Qwen3Config
from .qwen3 import Qwen3


class _Qwen3(Qwen3):
    # The checkpoint names Qwen3's weights from its root, so the decoder that
    # holds it adds no prefix.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class Qwen3Decoder(Module):
    """Dense Qwen3, registered for the ``DECODER`` role.

    It runs on one device, tensor-parallel across several, or with
    data-parallel attention, as the config says. Its embedding and head
    allreduce, so it takes signal buffers on every deployment.
    """

    graph_name = "qwen3"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    supports_data_parallel = True
    always_signal_buffers = True
    override_quantization_encoding = True
    data_parallel_batch_inputs = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "host_input_row_offsets",
        "data_parallel_splits",
    )
    """The data-parallel graph's batch, which predates the role contract
    and takes the host row offsets after the logit count."""

    def __init__(self, config: Qwen3Config) -> None:
        """Builds the model.

        Raises:
            NotImplementedError: If the config sets up expert parallelism,
                which registered decoders don't serve yet.
        """
        super().__init__()
        if config.ep_config is not None:
            raise NotImplementedError(
                "the registered Qwen3 decoder doesn't serve expert parallelism"
            )
        self.model = _Qwen3(config)
        self.kv_params = config.kv_params
        self._tie_word_embeddings = config.tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackArray | WeightData],
        *,
        override_quantization_encoding: bool = False,
        weight_alignment: int | None = None,
        strict: bool = True,
    ) -> None:
        """Loads the checkpoint, which may keep a tied head's weights."""
        super().load_state_dict(
            state_dict,
            override_quantization_encoding=override_quantization_encoding,
            weight_alignment=weight_alignment,
            strict=strict and not self._tie_word_embeddings,
        )

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model on the batch and returns its logits."""
        assert inputs.signals is not None
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(iter(tree.leaves(inputs.kv))),
            leaf=KVCacheInputsPerDevice,
        )
        return decoder_outputs(
            self.model(
                inputs.tokens,
                kv_collections,
                inputs.return_n_logits,
                inputs.input_row_offsets,
                list(inputs.signals),
                host_input_row_offsets=inputs.host_input_row_offsets,
                data_parallel_splits=inputs.data_parallel_splits,
            )
        )
