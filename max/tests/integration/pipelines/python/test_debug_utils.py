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
import torch
from max.driver.tensor import Tensor as MaxTensor
from pytest_mock import MockerFixture


def test_load_torch_intermediates(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    def fake_debug_model(*args: Any, output_path: Path, **kwargs: Any) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(2, 2), output_path / "layer0.out.pt")
        torch.save(torch.zeros(1), output_path / "layer1.out.pt")

    mocker.patch.object(
        dbg,
        "debug_model",
        autospec=True,
        side_effect=fake_debug_model,
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
    def fake_debug_model(*args: Any, output_path: Path, **kwargs: Any) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "node0-output.max").write_bytes(b"dummy")
        (output_path / "node1-output.max").write_bytes(b"dummy")

    def fake_load_max_tensor(path: PathLike[str]) -> MaxTensor:
        return MaxTensor.from_numpy(np.ones((3, 3), dtype=np.float32))

    mocker.patch.object(
        dbg,
        "debug_model",
        autospec=True,
        side_effect=fake_debug_model,
    )
    mocker.patch.object(
        dbg, "load_max_tensor", autospec=True, side_effect=fake_load_max_tensor
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
