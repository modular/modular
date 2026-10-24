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
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor, default_device


@module_dataclass
class LearnedScale(Module[[Tensor], Tensor]):
    """A layer that owns one learnable per-channel scale weight."""

    weight: Tensor
    """The learned per-channel scale of shape ``[dim]``."""

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


with default_device(CPU()):
    layer = LearnedScale(Tensor.ones([8]))
    scaled = layer(Tensor.ones([2, 8]))
    print([name for name, _ in layer.parameters])
    print(float(scaled.to_numpy()[0, 0]))
