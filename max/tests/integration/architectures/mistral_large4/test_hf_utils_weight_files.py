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
"""Weight-file discovery for a checkpoint sharded only as consolidated shards.

A Mistral-style checkpoint holds ``consolidated-00001-of-00005.safetensors``
beside a ``consolidated.safetensors.index.json`` and nothing else, which the
"ignore consolidated files when there is more than one weight" rule used to
filter out entirely. These tests pin the fallback and the rule it must not
weaken: consolidated shards nested in a training snapshot directory stay
ignored. CPU only; the files are empty stand-ins because discovery reads no
header.
"""

from __future__ import annotations

import glob
import types
from pathlib import Path

import pytest
from max.graph.weights import WeightsFormat
from max.pipelines.weights import hf_utils
from max.pipelines.weights.hf_utils import HuggingFaceRepo


def _touch(root: Path, *relative_paths: str) -> None:
    """Creates empty files, and the directories that hold them."""
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_weight_files_takes_top_level_consolidated_shards(
    tmp_path: Path,
) -> None:
    """A checkpoint with only consolidated shards loads them, in shard order."""
    _touch(
        tmp_path,
        "consolidated-00002-of-00003.safetensors",
        "consolidated-00001-of-00003.safetensors",
        "consolidated-00010-of-00003.safetensors",
        "consolidated.safetensors.index.json",
        "params.json",
    )
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    assert repo.weight_files[WeightsFormat.safetensors] == [
        "consolidated-00001-of-00003.safetensors",
        "consolidated-00002-of-00003.safetensors",
        # Sorted as text, the way the converter's own index orders them.
        "consolidated-00010-of-00003.safetensors",
    ]


def test_weight_files_orders_shards_independently_of_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery order is the filesystem's; the shard order must not be.

    ``glob`` hands back whatever order the filesystem enumerates in, which put
    ``model-00002`` ahead of ``model-00001`` on one machine and not on another.
    """
    _touch(
        tmp_path,
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    discover = glob.glob
    monkeypatch.setattr(
        hf_utils,
        "glob",
        types.SimpleNamespace(
            glob=lambda *args, **kwargs: sorted(
                discover(*args, **kwargs), reverse=True
            )
        ),
    )
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    assert repo.weight_files[WeightsFormat.safetensors] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]


def test_weight_files_ignores_consolidated_shards_in_a_training_snapshot(
    tmp_path: Path,
) -> None:
    """A snapshot's consolidated shards are repeated tensors, not the weights."""
    _touch(
        tmp_path,
        "checkpoint-100/consolidated-00001-of-00002.safetensors",
        "checkpoint-100/consolidated-00002-of-00002.safetensors",
        "checkpoint-200/consolidated-00001-of-00002.safetensors",
        "checkpoint-200/consolidated-00002-of-00002.safetensors",
    )
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    assert repo.weight_files[WeightsFormat.safetensors] == []


def test_weight_files_prefers_shards_over_consolidated_output(
    tmp_path: Path,
) -> None:
    """A repo shipping both carries every tensor twice; the shards win."""
    _touch(
        tmp_path,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "consolidated-00001-of-00001.safetensors",
        "checkpoint-50/model.safetensors",
    )
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    assert repo.weight_files[WeightsFormat.safetensors] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_weight_files_judges_consolidated_on_the_file_name(
    tmp_path: Path,
) -> None:
    """A directory called ``consolidated-*`` is not a consolidated shard.

    The rule reads the file's own name. On the path it would have matched this
    checkout's directory, sent both shards to the fallback with the
    consolidated output beside them, and loaded every tensor twice -- the one
    case where the fallback changes an outcome the shards already answered.
    """
    _touch(
        tmp_path,
        "consolidated-export/model-00001-of-00002.safetensors",
        "consolidated-export/model-00002-of-00002.safetensors",
        "consolidated-export/consolidated-00001-of-00001.safetensors",
    )
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    assert repo.weight_files[WeightsFormat.safetensors] == [
        "consolidated-export/model-00001-of-00002.safetensors",
        "consolidated-export/model-00002-of-00002.safetensors",
    ]
