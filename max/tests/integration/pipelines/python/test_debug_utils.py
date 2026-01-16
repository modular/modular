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

from os import PathLike
from pathlib import Path
from typing import Any

import max.tests.integration.tools.debugging_utils as dbg
import numpy as np
import pytest
import torch
from max.driver.buffer import Buffer as MaxBuffer
from pytest_mock import MockerFixture


def test_load_torch_intermediates(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(2, 2), output_path / "layer0.out.pt")
        torch.save(torch.zeros(1), output_path / "layer1.out.pt")

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    tensors = dbg.load_intermediate_tensors(
        model="dummy/model",
        framework="torch",
        output_dir=tmp_path,
        device_type="cpu",
        encoding_name=None,
    )
    assert set(tensors.keys()) == {"layer0.out.pt", "layer1.out.pt"}
    assert isinstance(tensors["layer0.out.pt"], torch.Tensor)
    assert isinstance(tensors["layer1.out.pt"], torch.Tensor)


def test_load_max_intermediates(tmp_path: Path, mocker: MockerFixture) -> None:
    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "node0-output.max").write_bytes(b"dummy")
        (output_path / "node1-output.max").write_bytes(b"dummy")

    def fake_load_max_buffer(path: PathLike[str]) -> MaxBuffer:
        return MaxBuffer.from_numpy(np.ones((3, 3), dtype=np.float32))

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )
    mocker.patch.object(
        dbg, "load_max_buffer", autospec=True, side_effect=fake_load_max_buffer
    )

    tensors = dbg.load_intermediate_tensors(
        model="dummy/model",
        framework="max",
        output_dir=tmp_path,
        device_type="cpu",
        encoding_name=None,
    )
    assert set(tensors.keys()) == {"node0-output.max", "node1-output.max"}
    for t in tensors.values():
        assert isinstance(t, torch.Tensor)
        assert tuple(t.shape) == (3, 3)


def test_get_torch_testdata_with_module(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test get_torch_testdata retrieves input and output tensors for a module."""

    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        # Create tensors following TorchPrintHook naming pattern
        # For module "encoder.layer.0", the full name is "model.encoder.layer.0"
        input_tensor = torch.randn(2, 4)
        output_tensor = torch.randn(2, 8)
        torch.save(input_tensor, output_path / "model.encoder.layer.0.input.pt")
        torch.save(
            output_tensor, output_path / "model.encoder.layer.0.output.pt"
        )

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    input_t, output_t = dbg.get_torch_testdata(
        model="dummy/model",
        module_name="encoder.layer.0",
        output_dir=tmp_path,
        device_type="cpu",
        encoding_name=None,
    )

    assert isinstance(input_t, torch.Tensor)
    assert isinstance(output_t, torch.Tensor)
    assert input_t.shape == (2, 4)
    assert output_t.shape == (2, 8)


def test_get_torch_testdata_top_level_model(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test get_torch_testdata with empty module_name for top-level model."""

    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        # For top-level model, name is just "model"
        input_tensor = torch.ones(1, 3)
        output_tensor = torch.zeros(1, 5)
        torch.save(input_tensor, output_path / "model.input.pt")
        torch.save(output_tensor, output_path / "model.output.pt")

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    input_t, output_t = dbg.get_torch_testdata(
        model="dummy/model",
        module_name="",
        output_dir=tmp_path,
    )

    assert input_t.shape == (1, 3)
    assert output_t.shape == (1, 5)


def test_get_torch_testdata_alternative_naming(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test get_torch_testdata with alternative tensor naming patterns."""

    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        # Use alternative naming: .args.0 for input, no suffix for output
        input_tensor = torch.randn(3, 3, 3)
        output_tensor = torch.randn(3, 3, 3)
        torch.save(input_tensor, output_path / "model.decoder.args.0.pt")
        torch.save(output_tensor, output_path / "model.decoder.pt")

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    input_t, output_t = dbg.get_torch_testdata(
        model="dummy/model",
        module_name="decoder",
        output_dir=tmp_path,
    )

    assert input_t.shape == (3, 3, 3)
    assert output_t.shape == (3, 3, 3)


def test_get_torch_testdata_missing_output(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test get_torch_testdata raises KeyError when output tensor is missing."""

    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        # Only create input tensor, no output
        input_tensor = torch.randn(2, 2)
        torch.save(input_tensor, output_path / "model.layer.input.pt")

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    with pytest.raises(KeyError, match=r"Output tensor for module.*not found"):
        dbg.get_torch_testdata(
            model="dummy/model",
            module_name="layer",
            output_dir=tmp_path,
        )


def test_get_torch_testdata_missing_input(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test get_torch_testdata raises KeyError when input tensor is missing."""

    def fake_run_debug_model(
        *args: Any, output_path: Path, **kwargs: Any
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        # Only create output tensor, no input
        output_tensor = torch.randn(2, 2)
        torch.save(output_tensor, output_path / "model.layer.output.pt")

    mocker.patch.object(
        dbg,
        "run_debug_model",
        autospec=True,
        side_effect=fake_run_debug_model,
    )

    with pytest.raises(KeyError, match=r"Input tensor for module.*not found"):
        dbg.get_torch_testdata(
            model="dummy/model",
            module_name="layer",
            output_dir=tmp_path,
        )
