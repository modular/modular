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

"""Implements the Orion nn.model."""

from __future__ import annotations

from max.nn.norm import LayerNorm
from max.pipelines.architectures.llama3.llama3 import Llama3

from .model_config import OrionConfig


class Orion(Llama3):
    """Llama3 graph with Orion's learned LayerNorm at every norm site.

    Orion is Llama with one substitution: it normalizes with
    ``nn.LayerNorm`` -- mean-centred, with a learned scale *and* bias -- where
    Llama uses ``RMSNorm``. The donor's ``norm_method="layer_norm"`` switch
    builds ``ConstantLayerNorm``, whose affines are hardcoded to ones/zeros and
    never bound to checkpoint weights, so the norms are replaced after
    construction instead.
    """

    def __init__(self, config: OrionConfig) -> None:
        super().__init__(config)

        def make_norm() -> LayerNorm:
            return LayerNorm(
                config.hidden_size,
                config.devices,
                config.norm_dtype or config.dtype,
                eps=config.layer_norm_eps,
                use_bias=True,
            )

        for block in self.layers:
            block.input_layernorm = make_norm()
            block.post_attention_layernorm = make_norm()
        self.norm = make_norm()
