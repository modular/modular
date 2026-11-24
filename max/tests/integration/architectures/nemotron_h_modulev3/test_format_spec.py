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

"""Tests for the FormatSpec the ModuleV3 Nemotron-H arch names per checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from max.pipelines.architectures.nemotron_h_modulev3 import arch
from max.pipelines.architectures.nemotron_h_modulev3.arch import (
    nemotron_format_spec,
)
from max.pipelines.weights.hf_utils import HuggingFaceRepo

NEMOTRON_3_5 = Path(arch.__file__).parent / "formatspec.json"


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", NEMOTRON_3_5),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", NEMOTRON_3_5),
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8", None),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", None),
    ],
)
def test_only_lightning_checkpoints_get_the_nemotron_3_5_spec(
    tmp_path: Path, checkpoint: str, expected: Path | None
) -> None:
    repo_dir = tmp_path / checkpoint
    repo_dir.mkdir(parents=True)
    document = nemotron_format_spec(HuggingFaceRepo(str(repo_dir)))
    assert document == expected
    assert document is None or document.is_file()
