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
# DOC: develop/layers.mdx

from max.driver import CPU
from max.experimental.nn import Linear, Module, ModuleList, RMSNorm
from max.experimental.tensor import Tensor, default_device


class NormalizedProjection(Module[[Tensor], Tensor]):
    """A linear projection whose output is normalized with RMSNorm."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = Linear(in_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class DecoderBlock(Module[[Tensor], Tensor]):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qk_proj = NormalizedProjection(dim, dim)  # your custom layer
        self.o_proj = Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(self.qk_proj(x))


class MiniDecoder(Module[[Tensor], Tensor]):
    """Two decoder blocks and an output projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.blocks = ModuleList([DecoderBlock(dim), DecoderBlock(dim)])
        self.lm_head = Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


with default_device(CPU()):
    model = MiniDecoder(dim=32)

    # The composed parameter names mirror the attribute hierarchy.
    for name in sorted(name for name, _ in model.parameters):
        print(name)
