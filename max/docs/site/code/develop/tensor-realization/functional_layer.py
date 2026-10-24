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
# DOC: develop/tensor-realization.mdx

from max.experimental import functional as F
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor


@module_dataclass
class MyLayer(Module[[Tensor], Tensor]):
    weight: Tensor
    bias: Tensor

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias


layer = MyLayer(weight=Tensor.ones([3, 4]), bias=Tensor.zeros([3]))
y = layer(Tensor.ones([2, 4]))
print(y.shape)
