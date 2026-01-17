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

"""Tests for compare_buffers CLI tool."""

from __future__ import annotations

from pathlib import Path

import compare_buffers
import pytest
import torch
from click.testing import CliRunner
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.type import DType


@pytest.fixture(scope="module")
def inference_session() -> InferenceSession:
    """Create a reusable inference session for all tests."""
    return InferenceSession(devices=[CPU()])


@pytest.fixture
def temp_directories(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary directories for torch and max tensors."""
    torch_dir = tmp_path / "torch"
    max_dir = tmp_path / "max"
    torch_dir.mkdir()
    max_dir.mkdir()
    return torch_dir, max_dir


def save_max_tensor(
    tensor: torch.Tensor,
    output_path: Path,
    tensor_name: str,
    session: InferenceSession,
) -> None:
    """Save a tensor in MAX checkpoint format using a simple graph.

    Args:
        tensor: Torch tensor to save.
        output_path: Directory where the .max file will be saved.
        tensor_name: Name for the tensor (without .max extension).
        session: Reusable inference session.
    """
    from max.graph import TensorValue

    def print_tensor(x: TensorValue) -> TensorValue:
        x.print(tensor_name)
        return x

    max_tensor = Buffer.from_dlpack(tensor)

    graph = Graph(
        f"save_{tensor_name}",
        print_tensor,
        [TensorType(DType.float32, tensor.shape, device=DeviceRef.CPU())],
    )

    session.set_debug_print_options(
        "BINARY_MAX_CHECKPOINT", output_directory=output_path
    )

    compiled_model = session.load(graph)
    _ = compiled_model.execute(max_tensor)


def test_cli_specific_tensors(
    temp_directories: tuple[Path, Path], inference_session: InferenceSession
) -> None:
    """Test CLI with specific tensor files."""
    torch_dir, max_dir = temp_directories

    tensor = torch.tensor([1.0, 2.0, 3.0])
    torch_file = torch_dir / "test_tensor.pt"
    torch.save(tensor, torch_file)

    save_max_tensor(tensor, max_dir, "test_tensor", inference_session)
    max_file = max_dir / "test_tensor.max"

    runner = CliRunner()
    result = runner.invoke(
        compare_buffers.main,
        [
            "--torch-tensor",
            str(torch_file),
            "--max-tensor",
            str(max_file),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Comparing tensors:" in result.output
    assert "No tolerances provided - reporting metrics only" in result.output


def test_cli_auto_match(
    temp_directories: tuple[Path, Path], inference_session: InferenceSession
) -> None:
    """Test CLI with automatic tensor matching from directories."""
    torch_dir, max_dir = temp_directories

    tensor1 = torch.tensor([1.0, 2.0])
    tensor2 = torch.tensor([3.0, 4.0])

    torch.save(tensor1, torch_dir / "tensor1.pt")
    torch.save(tensor2, torch_dir / "tensor2.pt")

    save_max_tensor(tensor1, max_dir, "tensor1", inference_session)
    save_max_tensor(tensor2, max_dir, "tensor2", inference_session)

    runner = CliRunner()
    result = runner.invoke(
        compare_buffers.main,
        [
            "--torch-tensor",
            str(torch_dir),
            "--max-tensor",
            str(max_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Found 2 matching tensor pair(s)" in result.output
    assert "Comparison complete for 2 tensor(s)" in result.output


def test_cli_custom_tolerances(
    temp_directories: tuple[Path, Path], inference_session: InferenceSession
) -> None:
    """Test CLI with custom rtol and atol values."""
    torch_dir, max_dir = temp_directories

    tensor1 = torch.tensor([1.0, 2.0])
    tensor2 = torch.tensor([1.01, 2.01])  # 1% difference

    torch_file = torch_dir / "test.pt"
    torch.save(tensor1, torch_file)

    save_max_tensor(tensor2, max_dir, "test", inference_session)
    max_file = max_dir / "test.max"

    runner = CliRunner()
    result = runner.invoke(
        compare_buffers.main,
        [
            "--torch-tensor",
            str(torch_file),
            "--max-tensor",
            str(max_file),
            "--rtol",
            "0.02",  # 2% tolerance
            "--atol",
            "0.02",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Using rtol: 0.02, atol: 0.02" in result.output
    assert "close" in result.output.lower()
