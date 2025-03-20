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

from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, TensorShape


# This example highlights the very basic API structure around building a MAX
# Graph model and executing it through the MAX engine APIS.
# Simply run this mojo file to create, load, and execute this simple model.
def main():
    graph = Graph(TensorType(DType.float32, 2, 6))

    # Create a constant for usage in the matmul op below:
    matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
    matmul_constant = graph.constant(matmul_constant_value)

    # Start adding a sequence of operator calls to build the graph.
    # We can use the subscript notation to get the graph's first input tensor:
    matmul = graph[0] @ matmul_constant
    relu = ops.relu(matmul)
    softmax = ops.softmax(relu)
    graph.output(softmax)

    # Load the graph:
    session = InferenceSession()
    model = session.load(graph)

    # Print the input/output names:
    in_names = model.get_model_input_names()
    for name in in_names:
        print("Input:", name[])
    out_names = model.get_model_output_names()
    for name in out_names:
        print("Output:", name[])

    # Execute the model:
    input = Tensor[DType.float32](TensorShape(2, 6), 0.5)
    results = model.execute("input0", input^)
    output = results.get[DType.float32]("output0")
    print(output)
