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

"""Generate MEF files for testing symbolic dimensions in C API."""

import click
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType, ops


@click.group()
@click.argument("output_path")
@click.pass_context
def main(ctx: click.Context, output_path: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["output_path"] = output_path


def write_mef(ctx: click.Context, graph: Graph) -> None:
    """Compiles the graph and writes the compiled MEF to a file."""
    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
    model._export_mef(ctx.obj["output_path"])


@main.command()
@click.pass_context
def static_dimensions(ctx: click.Context) -> None:
    """Generates MEF with static dimensions [8, 10]."""
    input_type = TensorType(
        dtype=DType.float32, shape=(8, 10), device=DeviceRef.CPU()
    )

    with Graph("static_add", input_types=(input_type,)) as graph:
        output = graph.inputs[0].tensor + 1.0
        graph.output(output)

    write_mef(ctx, graph)


@main.command()
@click.pass_context
def symbolic_dimensions(ctx: click.Context) -> None:
    """Generates MEF with symbolic batch dimension ["batch", 10]."""
    input_type = TensorType(
        dtype=DType.float32, shape=("batch", 10), device=DeviceRef.CPU()
    )

    with Graph("symbolic_add", input_types=(input_type,)) as graph:
        output = graph.inputs[0].tensor * 2.0
        graph.output(output)

    write_mef(ctx, graph)


@main.command()
@click.pass_context
def same_symbolic(ctx: click.Context) -> None:
    """Generates MEF with two inputs sharing same symbolic dimension ["seq_len", 768]."""
    input_type = TensorType(
        dtype=DType.float32, shape=("seq_len", 768), device=DeviceRef.CPU()
    )

    with Graph("same_symbolic", input_types=(input_type, input_type)) as graph:
        result = graph.inputs[0].tensor + graph.inputs[1].tensor
        graph.output(result)

    write_mef(ctx, graph)


@main.command()
@click.pass_context
def algebraic_dimensions(ctx: click.Context) -> None:
    """Generates MEF that reshapes input [batch, dim] -> [batch, 4, dim // 4]."""
    input_type = TensorType(
        dtype=DType.float32, shape=("batch", "dim"), device=DeviceRef.CPU()
    )

    with Graph("algebraic", input_types=(input_type,)) as graph:
        x = graph.inputs[0].tensor
        x = x.rebind(["batch", 4 * (Dim("dim") // 4)])
        reshaped = ops.reshape(x, (Dim("batch"), 4, Dim("dim") // 4))
        graph.output(reshaped)

    write_mef(ctx, graph)


if __name__ == "__main__":
    main()
