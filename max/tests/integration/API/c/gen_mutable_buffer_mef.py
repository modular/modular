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

"""Generate MEF files for testing mutable buffers in C API."""

import click
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph


@click.command()
@click.argument("output_path")
def main(output_path: str) -> None:
    buffer_type = BufferType(
        dtype=DType.float32, shape=[10], device=DeviceRef.CPU()
    )

    with Graph("mutate_cpu", input_types=(buffer_type,)) as graph:
        buffer = graph.inputs[0].buffer
        # Load from buffer, modify, store back
        tensor = buffer[...]
        buffer[...] = tensor * 2.0 + 1.0
        graph.output()

    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
    model._export_mef(output_path)


if __name__ == "__main__":
    main()
