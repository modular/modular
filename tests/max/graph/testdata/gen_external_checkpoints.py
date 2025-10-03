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
"""Generates external checkpoints for testing weight loading.

br //SDK/lib/API/python/tests/graph/testdata:gen_external_checkpoints -- $(pwd)/SDK/lib/API/python/tests/graph/testdata/
"""

from pathlib import Path

import click
import numpy as np
import safetensors.torch as safe_torch
import torch
from gguf import GGMLQuantizationType, GGUFWriter


def test_data():
    return {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3]),
    }


def write_gguf(filename: Path) -> None:
    gguf_writer = GGUFWriter(str(filename), "example")

    data: dict[str, np.ndarray] = {
        "a": np.arange(10, dtype=np.int32).reshape(5, 2),
        "b": np.full((1, 2, 3), 3.5, dtype=np.float64),
        "c": np.array(5432.1, dtype=np.float32),
        "fancy/name": np.array([1, 2, 3], dtype=np.int64),
    }
    for key, tensor in data.items():
        gguf_writer.add_tensor(key, tensor)

    # Separately add Bfloat16 tensor.
    gguf_writer.add_tensor(
        "bf16",
        torch.tensor([123, 45], dtype=torch.bfloat16)
        .view(torch.float16)
        .numpy(),
        raw_dtype=GGMLQuantizationType.BF16,
    )

    # Add a quantized tensor (fake data).
    gguf_writer.add_tensor(
        "quantized",
        np.arange(0, 288, dtype=np.uint8).reshape(2, 144),
        raw_dtype=GGMLQuantizationType.Q4_K,
    )

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


def write_pytorch(filename: Path) -> None:
    data = {
        "a": torch.arange(10, dtype=torch.int32).reshape(5, 2),
        "b": torch.full((1, 2, 3), 3.5, dtype=torch.float64),
        "c": torch.tensor(5432.1, dtype=torch.float32),
        "fancy/name": torch.tensor([1, 2, 3], dtype=torch.int64),
        "bf16": torch.tensor([123, 45], dtype=torch.bfloat16),
    }

    torch.save(data, filename)


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
    write_pytorch(output_directory / "example_data.pt")
    write_gguf(output_directory / "example_data.gguf")
    write_safetensors(output_directory / "example_data")


if __name__ == "__main__":
    main()
