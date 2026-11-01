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

"""Tests for header-only (weightless) safetensors stubs.

``--use-dummy-weights`` lets a compile-only run skip the checkpoint
download: compilation reads only tensor names, shapes and dtypes, all of
which a safetensors header carries. These tests pin that a stub is
indistinguishable from the real file to every reader a compile goes
through, and that the flag reaches the config that acts on it.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from max._core.safetensors import safe_open
from max.graph.weights import load_weights
from max.pipelines.lib import PipelineArgs
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.weights import hf_utils

_METADATA = {"format": "pt", "quantization": "nvfp4"}


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Writes a real safetensors file carrying a ``__metadata__`` section."""
    header: dict[str, object] = {"__metadata__": _METADATA}
    buffers: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        contiguous = np.ascontiguousarray(array)
        header[name] = {
            "dtype": "F32",
            "shape": list(contiguous.shape),
            "data_offsets": [offset, offset + contiguous.nbytes],
        }
        buffers.append(contiguous.tobytes())
        offset += contiguous.nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for buffer in buffers:
            f.write(buffer)


def _leading_header_bytes(path: Path) -> bytes:
    """Returns a file's first ``8 + header_length`` bytes.

    Stands in for what ``_fetch_safetensors_header`` pulls over HTTP.
    """
    with open(path, "rb") as f:
        prefix = f.read(8)
        header_length: int = struct.unpack("<Q", prefix)[0]
        return prefix + f.read(header_length)


@pytest.fixture
def real_checkpoint(tmp_path: Path) -> Path:
    """A real safetensors file big enough for sparseness to be measurable."""
    path = tmp_path / "real.safetensors"
    _write_safetensors(
        path,
        {
            "model.layers.0.mlp.gate_proj.weight": np.arange(
                256 * 1024, dtype=np.float32
            ).reshape(256, 1024),
            "model.layers.0.mlp.down_proj.weight": np.ones(
                (1024, 256), dtype=np.float32
            ),
        },
    )
    return path


def test_stub_stands_in_for_the_real_checkpoint(
    tmp_path: Path, real_checkpoint: Path
) -> None:
    stub = tmp_path / "stub.safetensors"
    hf_utils._write_header_only_safetensors(
        stub, _leading_header_bytes(real_checkpoint)
    )

    real_stat = real_checkpoint.stat()
    stub_stat = stub.stat()
    assert stub_stat.st_size == real_stat.st_size
    # The data region is a hole, so the stub occupies a small fraction of the
    # real file's blocks. A loose bound keeps this from depending on the
    # filesystem's block size.
    assert stub_stat.st_blocks < real_stat.st_blocks // 4

    real_weights = load_weights([real_checkpoint])
    stub_weights = load_weights([stub])
    real_data = {
        name: (weight.data().dtype, tuple(weight.data().shape))
        for name, weight in real_weights.items()
    }
    stub_data = {
        name: (weight.data().dtype, tuple(weight.data().shape))
        for name, weight in stub_weights.items()
    }
    assert stub_data == real_data

    with safe_open(stub) as handle:
        assert handle.metadata() == _METADATA

    gate_proj = stub_weights.model.layers[0].mlp.gate_proj.weight.data()
    assert not np.from_dlpack(gate_proj).any()


def test_header_only_weight_files_writes_stubs_once(
    tmp_path: Path, real_checkpoint: Path
) -> None:
    header = _leading_header_bytes(real_checkpoint)
    filenames = [
        "model-00001-of-00002.safetensors",
        "transformer/model-00002-of-00002.safetensors",
    ]
    fetched: list[str] = []

    def fake_fetch(repo_id: str, filename: str, revision: str) -> bytes:
        fetched.append(filename)
        return header

    with (
        patch.object(
            hf_utils, "_header_only_weights_dir", return_value=tmp_path / "hdr"
        ),
        patch.object(hf_utils, "_resolve_commit_sha", return_value="cafef00d"),
        patch.object(
            hf_utils, "_fetch_safetensors_header", side_effect=fake_fetch
        ),
    ):
        paths = hf_utils.header_only_weight_files("org/name", filenames)
        repeat = hf_utils.header_only_weight_files("org/name", filenames)

    expected_root = tmp_path / "hdr" / "models--org--name" / "cafef00d"
    assert paths == [expected_root / filename for filename in filenames]
    assert all(path.is_file() for path in paths)
    assert repeat == paths
    # Every stub was already on disk the second time round.
    assert sorted(fetched) == sorted(filenames)


def test_header_only_weight_files_rejects_gguf(tmp_path: Path) -> None:
    with (
        patch.object(
            hf_utils, "_header_only_weights_dir", return_value=tmp_path
        ),
        pytest.raises(ValueError, match="only support safetensors"),
    ):
        hf_utils.header_only_weight_files("org/name", ["model.gguf"])


@pytest.fixture
def online_config() -> Iterator[MAXModelConfig]:
    """A header-only config for an online repo, with no network access.

    ``MAXModelConfig`` builds its weight repo handle on demand, and the
    handle's constructor checks repo access, so the patches have to outlive
    construction. CI runs with ``HF_HUB_OFFLINE=1``, under which the handle
    resolves a repo through the local cache instead and a placeholder repo
    does not exist there, so offline mode is switched off for the fixture.
    """
    with (
        patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
    ):
        config = MAXModelConfig(
            model_path="org/model",
            weight_path=[Path("model.safetensors")],
            header_only_weights=True,
        )
        assert config.huggingface_weight_repo.repo_type == "online"
        yield config


def test_resolved_weight_paths_requires_compile_only_mode(
    online_config: MAXModelConfig,
) -> None:
    with (
        patch(
            "max.pipelines.lib.config.model_config.is_virtual_device_mode",
            return_value=False,
        ),
        pytest.raises(ValueError, match="compile-only mode"),
    ):
        online_config.resolved_weight_paths()


def test_resolved_weight_paths_uses_stubs_instead_of_downloading(
    online_config: MAXModelConfig,
) -> None:
    stub = Path("/stubs/models--org--model/sha/model.safetensors")
    with (
        patch(
            "max.pipelines.lib.config.model_config.is_virtual_device_mode",
            return_value=True,
        ),
        patch(
            "max.pipelines.lib.config.model_config.header_only_weight_files",
            return_value=[stub],
        ) as mock_header_only,
        patch(
            "max.pipelines.lib.config.model_config.download_weight_files"
        ) as mock_download,
    ):
        assert online_config.resolved_weight_paths() == [stub]
    mock_download.assert_not_called()
    assert mock_header_only.call_args.kwargs == {
        "huggingface_model_id": "org/model",
        "filenames": ["model.safetensors"],
        "revision": online_config.huggingface_weight_repo.revision,
    }


def test_flag_reaches_the_model_config() -> None:
    args = PipelineArgs.from_flat_kwargs(header_only_weights=True)
    assert args.header_only_weights is True
    assert MAXModelConfig.from_pipeline_args(args).header_only_weights is True
