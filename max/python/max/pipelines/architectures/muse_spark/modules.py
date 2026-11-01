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
"""Registered Muse Spark text computation."""

from __future__ import annotations

from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype

from .model_config import MuseSparkConfig
from .muse_spark import MuseSpark, compute_dtype


class MuseSparkDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs Spark's text tower without a dummy vision graph input."""

    graph_name = "MuseSpark"

    def __init__(self, config: MuseSparkConfig) -> None:
        super().__init__()
        assert config.mesh is not None
        self.mesh = config.mesh
        self.hidden_size = config.text_config.hidden_size
        self.compute_dtype = compute_dtype(config)
        with default_dtype(self.compute_dtype):
            self.model = MuseSpark(config, self.mesh)
        self.model.name_transparent = True

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the existing text computation with the registered KV tree."""
        empty_embeddings = F.broadcast_to(
            F.constant(0, self.compute_dtype, device=self.mesh),
            [0, self.hidden_size],
        )
        empty_indices = F.broadcast_to(
            F.constant(0, DType.int32, device=self.mesh), [0]
        )
        outputs = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            empty_embeddings,
            empty_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=outputs[0],
            logits=outputs[1] if len(outputs) == 3 else None,
            logit_offsets=outputs[2] if len(outputs) == 3 else None,
        )
