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

from max import engine
from max.graph import Graph, TensorType, Type
from max.tensor import Tensor


def main():
    # 1. define add graph
    graph = Graph(
        in_types=List[Type](
            TensorType(DType.float32, 1), TensorType(DType.float32, 1)
        )
    )
    out = graph[0] + graph[1]
    graph.output(out)
    graph.verify()
    print("finall graph:", graph)
    # 2. load and compile the graph
    session = engine.InferenceSession()
    model = session.load(graph)
    print("input names are:")
    for input_name in model.get_model_input_names():
        print(input_name[])

    # 3. Execute / run the graph with some inputs
    print("set some input values:")
    input0 = Tensor[DType.float32](List[Float32](1.0))
    print("input0:", input0)
    input1 = Tensor[DType.float32](List[Float32](1.0))
    print("input1:", input1)
    ret = model.execute("input0", input0^, "input1", input1^)
    print("result:", ret.get[DType.float32]("output0"))
