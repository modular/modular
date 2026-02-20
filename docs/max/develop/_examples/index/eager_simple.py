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
# DOC: max/develop/index.mdx

from max import functional as F
from max.driver import CPU
from max.tensor import Tensor

# Create tensor from Python data
x = Tensor.constant([1.0, -2.0, 3.0, -4.0, 5.0], device=CPU())

y = F.relu(x)

# Results are available right away
print(y)
