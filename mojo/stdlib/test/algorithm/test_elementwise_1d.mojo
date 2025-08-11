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

from math import erf, exp, tanh
from sys.info import simdwidthof

from algorithm import elementwise
from buffer import NDBuffer

from utils.index import IndexList
from testing import assert_almost_equal


def test_elementwise_1d():
    alias num_elements = 64
    var ptr = UnsafePointer[Float32].alloc(num_elements)

    var vector = NDBuffer[DType.float32, 1, _, num_elements](ptr)

    for i in range(len(vector)):
        vector[i] = i

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var elem = vector.load[width=simd_width](idx[0])
        var val = exp(erf(tanh(elem + 1)))
        vector.store[width=simd_width](idx[0], val)

    elementwise[func, simdwidthof[DType.float32]()](IndexList[1](num_elements))

    assert_almost_equal(vector[0], 2.051446)

    ptr.free()


def main():
    test_elementwise_1d()
