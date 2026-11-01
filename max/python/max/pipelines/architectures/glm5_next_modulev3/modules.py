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
"""Registered GLM-5.3-Flash decoder."""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn.legacy.comm.ep import EPBatchManager
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype

from .glm5_next import Glm5Next
from .model_config import Glm5NextConfig


class Glm5NextDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Binds GLM's existing sparse-MLA and KDA computation to registered inputs."""

    graph_name = "Glm5Next"

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        ep = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.model = Glm5Next(config, config.kv_params, ep)
        self.model.name_transparent = True

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the decoder with its KV state and optional expert exchange."""
        assert inputs.batch_context_lengths is not None
        ep = () if inputs.ep is None else tuple(tree.leaves(inputs.ep))
        return self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.batch_context_lengths[0],
            *tree.leaves(inputs.kv),
            *ep,
        )
