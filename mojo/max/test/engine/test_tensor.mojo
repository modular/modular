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
# RUN: %mojo -debug-level full %s

from max.engine import EngineTensorView, InferenceSession
from max.tensor import Tensor, TensorShape
from testing import assert_equal

from utils.index import Index


fn test_tensor_view() raises:
    var t1 = Tensor[DType.float32](TensorShape(3), 1.0, 2.0, 3.0)

    var t1_view = EngineTensorView(t1)

    assert_equal(t1.unsafe_ptr(), t1_view.data[DType.float32]())

    assert_equal(t1.spec(), t1_view.spec())


fn test_tensor_value() raises:
    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    var session = InferenceSession()
    var value = session.new_borrowed_tensor_value(t1)

    var t2 = value.as_tensor_copy[DType.float32]()

    assert_equal(t1, t2)


fn main() raises:
    test_tensor_view()
    test_tensor_value()
