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
# DOC: develop/broadcasting.mdx

# fix_rank.py: use unsqueeze to fix a rank mismatch before broadcasting
#
# Output:
#   [Dim(4), Dim(8)]

from max.experimental.tensor import Tensor

x = Tensor.ones([4, 8])  # (batch=4, features=8)
scale = Tensor([1.0, 2.0, 3.0, 4.0])  # (batch=4,)
scale2d = scale.unsqueeze(-1)  # (batch=4, 1)
result = x * scale2d
print(result.shape)
