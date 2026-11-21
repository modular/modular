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
"""Generates external checkpoints for testing weight loading.

br //max/tests/tests/graph/testdata:gen_external_checkpoints -- $(pwd)/max/tests/tests/graph/testdata/
"""

from pathlib import Path

import click
import numpy as np
import safetensors.torch as safe_torch
import torch
from max._core.gguf import GGUFWriter

# GGML tensor type ids, as recorded in a GGUF header.
_GGML_F32 = 0
_GGML_Q4_K = 12
_GGML_I32 = 26
_GGML_I64 = 27
_GGML_F64 = 28
_GGML_BF16 = 30

_GGML_TYPE_OF_NUMPY = {
    np.dtype(np.float32): _GGML_F32,
    np.dtype(np.int32): _GGML_I32,
    np.dtype(np.int64): _GGML_I64,
    np.dtype(np.float64): _GGML_F64,
}


def test_data():  # noqa: ANN201
    return {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3]),
    }


def write_gguf(filename: Path) -> None:
    writer = GGUFWriter()
    writer.set_string("general.architecture", "example")

    data: dict[str, np.ndarray] = {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3], dtype=np.int64),
    }
    for key, tensor in data.items():
        writer.add_tensor(
            key,
            _GGML_TYPE_OF_NUMPY[tensor.dtype],
            list(tensor.shape),
            tensor.tobytes(),
        )

    # Separately add a bfloat16 tensor: GGUF stores its 16-bit patterns.
    bf16 = (
        np.array([123.0, 45.0], dtype=np.float32).view(np.uint32) >> 16
    ).astype(np.uint16)
    writer.add_tensor("bf16", _GGML_BF16, [2], bf16.tobytes())

    # Add a Q4_K tensor of fake data: two blocks of 256 elements, 144 bytes
    # each.
    writer.add_tensor(
        "quantized",
        _GGML_Q4_K,
        [2, 256],
        np.arange(0, 288, dtype=np.uint8).tobytes(),
    )

    writer.write(filename)


def write_safetensors(filename_prefix: Path) -> None:
    for i in range(1, 3):
        data = {
            f"{i}.a": torch.arange(10, dtype=torch.int32).reshape(5, 2),
            f"{i}.b": torch.full((1, 2, 3), 3.5, dtype=torch.float64),
            f"{i}.c": torch.tensor(5432.1, dtype=torch.float32),
            f"{i}.fancy/name": torch.tensor([1, 2, 3], dtype=torch.int64),
            f"{i}.bf16": torch.tensor([123, 45], dtype=torch.bfloat16),
            f"{i}.float8_e4m3fn": torch.tensor(
                [11.0, 250.0], dtype=torch.float8_e4m3fn
            ),
            f"{i}.float8_e5m2": torch.tensor(
                [13.0, 223.0], dtype=torch.float8_e5m2
            ),
        }

        safe_torch.save_file(data, f"{filename_prefix}_{i}.safetensors")


@click.command()
@click.argument("output_directory", type=Path)
def main(output_directory: Path) -> None:
    write_gguf(output_directory / "example_data.gguf")
    write_safetensors(output_directory / "example_data")


if __name__ == "__main__":
    main()
