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

# COM: Verify implict conversion between dtypes does not happen

# RUN: mkdir -p %t
# RUN: rm -rf %t/test-build-fail
# COM: Verify that this code does not compile successfully.
# RUN: not %mojo-build %s -o %t/test-build-fail

from max.driver import Tensor
from max.tensor import TensorShape


def main():
    var t: Tensor[DType.bool, 2]
    t = Tensor[DType.int64, 2](TensorShape(1, 2))
    print(t)
