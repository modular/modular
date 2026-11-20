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
"""The bias-free Kimi K3 patchmergerv2 projector."""

from __future__ import annotations

from max import functional as F
from max.nn import Linear, Module
from max.tensor import Tensor

from ...vision_config import KimiK3VisionConfig
from .norm import VisionRMSNorm


class PatchMergerMLP(Module[[Tensor], Tensor]):
    """Projects each spatial patch group and normalizes the output."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        self.input_dim = config.mm_hidden_size * (
            config.merge_kernel_size[0] * config.merge_kernel_size[1]
        )
        self.linear1 = Linear(self.input_dim, self.input_dim, bias=False)
        self.linear2 = Linear(
            self.input_dim, config.text_hidden_size, bias=False
        )
        self.post_norm = VisionRMSNorm(
            config.text_hidden_size, eps=config.projector_ln_eps
        )

    def forward(self, x: Tensor) -> Tensor:
        """Flattens four patches and applies the projector without input norm."""
        x = x.reshape([x.shape[0], self.input_dim])
        return self.post_norm(self.linear2(F.gelu(self.linear1(x))))
