# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""
You can import these APIs from the `max.tensor` package. For example:

```mojo
from max.tensor import RuntimeTensorSpec
```
"""

from sys import size_of


from utils import IndexList, product


@fieldwise_init
@register_passable("trivial")
struct RuntimeTensorSpec[dtype: DType, rank: Int](ImplicitlyCopyable, Movable):
    var shape: IndexList[rank]

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]

    fn bytecount(self) -> Int:
        """
        Gets the total byte count.

        Returns:
          The total byte count.
        """
        return product(self.shape) * size_of[dtype]()
