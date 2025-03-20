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

from random import seed

from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.tensor import Tensor, TensorShape


def main():
    # 1.Define the graph
    graph = Graph(TensorType(DType.float32, "m", 2))
    # create a constant tensor
    constant_value = Tensor[DType.float32](TensorShape(2, 2), 42.0)
    print("constant value:", constant_value)
    # create a constant node
    constant_symbol = graph.constant(constant_value)
    # create a matmul node
    mm = graph[0] @ constant_symbol
    graph.output(mm)
    # verify
    graph.verify()

    # 2. Load and compile the graph
    session = InferenceSession()
    model = session.load(graph)

    # 3. Execute the graph with inputs
    # generate random inputs
    seed(42)
    input0 = Tensor[DType.float32].randn((2, 2))
    print("random 2x2 input0:", input0)
    ret = model.execute("input0", input0^)
    print("matmul 2x2 result:", ret.get[DType.float32]("output0"))
    # with 3 x 2 matrix input
    input0 = Tensor[DType.float32].randn((3, 2))
    print("random 3x2 input0:", input0)
    ret = model.execute("input0", input0^)
    print("matmul 3x2 result:", ret.get[DType.float32]("output0"))
