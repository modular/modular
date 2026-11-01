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
"""The GPT OSS decoder registered for the ``DECODER`` role."""

from __future__ import annotations

from max import tree
from max.graph import TensorValue
from max.nn.legacy.layer import Module
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from .gpt_oss import GptOss
from .model_config import GptOssConfig


class _GptOss(GptOss):
    # The checkpoint names GPT OSS's weights from its root, so the decoder
    # that holds it adds no prefix.
    @property
    def _omit_module_attr_name(self) -> bool:
        return True


class GptOssDecoder(Module):
    """GPT OSS over a tensor-parallel group of devices, for ``DECODER``.

    Every device takes a signal buffer, even alone, and reads its own copy of
    the row offsets, which the graph makes from the one it takes.
    """

    graph_name = "gpt_oss"
    """The name the graph compiled under before registration (the checkpoint's
    ``model_type``), which keys precompiled MEFs."""

    always_signal_buffers = True

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.model = _GptOss(config)
        self.devices = config.devices
        self.kv_params = config.kv_params

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model on the batch and returns its logits."""
        assert inputs.signals is not None
        sliding_kv, global_kv = self.kv_params.unflatten_basic_kv_tree(
            iter(tree.leaves(inputs.kv))
        )
        return decoder_outputs(
            self.model(
                inputs.tokens,
                list(inputs.signals),
                sliding_kv,
                global_kv,
                inputs.return_n_logits,
                # Each device's layers read the row offsets on their own
                # device.
                [
                    inputs.input_row_offsets,
                    *(
                        inputs.input_row_offsets.to(device)
                        for device in self.devices[1:]
                    ),
                ],
            )
        )
