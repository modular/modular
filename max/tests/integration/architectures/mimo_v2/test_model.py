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
"""Tests which checkpoint tensors the MiMo-V2 model hands its adapter."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.mimo_v2.model import text_model_weights
from max.pipelines.weights import HuggingFaceRepo


def _write(path: Path, names: list[str]) -> Path:
    """Writes a safetensors file of one-element F32 tensors."""
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
        for i, name in enumerate(names)
    }
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(4 * len(names))
    )
    return path


def _index(directory: Path, names: list[str]) -> None:
    weight_map = {name: "model.safetensors" for name in names}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def _repo(tmp_path: Path) -> tuple[list[Path], list[str]]:
    """A repo with the text model at the root, as the export ships it, and
    a drafter's safetensors in a subdirectory."""
    text = ["model.norm.weight", "lm_head.weight"]
    drafter = ["layers.0.self_attn.q_proj.weight"]
    files = [
        _write(tmp_path / "model.safetensors", text),
        _write(tmp_path / "dflash" / "model.safetensors", drafter),
    ]
    return files, text


def test_the_index_drops_sidecar_tensors(tmp_path: Path) -> None:
    files, text = _repo(tmp_path)
    _index(tmp_path, text)

    got = text_model_weights(
        HuggingFaceRepo(str(tmp_path)), SafetensorWeights(files)
    )

    assert sorted(got) == sorted(text)


def test_the_index_is_read_from_the_subfolder(tmp_path: Path) -> None:
    files, text = _repo(tmp_path)
    _index(tmp_path, text)
    _index(tmp_path / "dflash", ["layers.0.self_attn.q_proj.weight"])

    got = text_model_weights(
        HuggingFaceRepo(str(tmp_path), subfolder="dflash"),
        SafetensorWeights(files),
    )

    assert sorted(got) == ["layers.0.self_attn.q_proj.weight"]


def test_without_an_index_every_tensor_is_kept(tmp_path: Path) -> None:
    files, text = _repo(tmp_path)

    got = text_model_weights(
        HuggingFaceRepo(str(tmp_path)), SafetensorWeights(files)
    )

    # The adapter then rejects the tensors it does not expect.
    assert sorted(got) == sorted([*text, "layers.0.self_attn.q_proj.weight"])
