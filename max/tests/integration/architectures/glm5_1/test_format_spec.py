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

"""Tests for the FormatSpec the GLM architectures name per checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from max.pipelines.architectures.glm5_1 import arch
from max.pipelines.architectures.glm5_1.arch import glm_format_spec
from max.pipelines.weights.hf_utils import HuggingFaceRepo

GLM5_3 = Path(arch.__file__).parent / "formatspec.json"


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("zai-org/GLM-5.3", GLM5_3),
        ("RadixArk/GLM-5.3-NVFP4", GLM5_3),
        ("zai-org/GLM-5.1", None),
        ("zai-org/GLM-5.2-FP8", None),
    ],
)
def test_only_glm_5_3_checkpoints_get_the_glm5_3_spec(
    tmp_path: Path, checkpoint: str, expected: Path | None
) -> None:
    repo_dir = tmp_path / checkpoint
    repo_dir.mkdir(parents=True)
    document = glm_format_spec(HuggingFaceRepo(str(repo_dir)))
    assert document == expected
    assert document is None or document.is_file()
