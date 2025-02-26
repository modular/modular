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

import os
from pathlib import Path

from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


def create_mandelbrot_graph(
    width: int,
    height: int,
    min_x: float,
    min_y: float,
    scale_x: float,
    scale_y: float,
    max_iterations: int,
) -> Graph:
    """Configure a graph to run a Mandelbrot kernel."""
    output_dtype = DType.int32
    with Graph(
        "mandelbrot",
    ) as graph:
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        result = ops.custom(
            name="mandelbrot",
            values=[
                ops.constant(min_x, dtype=DType.float32),
                ops.constant(min_y, dtype=DType.float32),
                ops.constant(scale_x, dtype=DType.float32),
                ops.constant(scale_y, dtype=DType.float32),
                ops.constant(max_iterations, dtype=DType.int32),
            ],
            out_types=[TensorType(dtype=output_dtype, shape=[height, width])],
        )[0].tensor

        # Return the result of the custom operation as the output of the graph.
        graph.output(result)
        return graph


if __name__ == "__main__":
    # This is necessary only in specific build environments.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    # Establish Mandelbrot set ranges.
    WIDTH = 15
    HEIGHT = 15
    MAX_ITERATIONS = 100
    MIN_X = -1.5
    MAX_X = 0.7
    MIN_Y = -1.12
    MAX_Y = 1.12

    # Configure our simple graph.
    scale_x = (MAX_X - MIN_X) / WIDTH
    scale_y = (MAX_Y - MIN_Y) / HEIGHT
    graph = create_mandelbrot_graph(
        WIDTH, HEIGHT, MIN_X, MIN_Y, scale_x, scale_y, MAX_ITERATIONS
    )

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Set up an inference session that runs the graph on a GPU, if available.
    session = InferenceSession(
        devices=[device],
        custom_extensions=path,
    )
    # Compile the graph.
    model = session.load(graph)

    # Perform the calculation on the target device.
    result = model.execute()[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Iterations to escape:")
    print(result.to_numpy())
    print()
