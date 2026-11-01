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

"""Tests for the FormatSpec the Laguna architectures name per checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from max.pipelines.architectures.laguna import arch
from max.pipelines.architectures.laguna.arch import laguna_format_spec
from max.pipelines.weights.hf_utils import HuggingFaceRepo

LAGUNA = Path(arch.__file__).parent / "formatspec.json"


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("poolside/Laguna-S-2.1-NVFP4", LAGUNA),
        ("poolside/Laguna-M.1-NVFP4", None),
    ],
)
def test_only_laguna_s_2_1_checkpoints_get_the_laguna_spec(
    tmp_path: Path, checkpoint: str, expected: Path | None
) -> None:
    repo_dir = tmp_path / checkpoint
    repo_dir.mkdir(parents=True)
    document = laguna_format_spec(HuggingFaceRepo(str(repo_dir)))
    assert document == expected
    assert document is None or document.is_file()
