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

import logging

import hf_repo_lock
import pytest
from max.entrypoints import pipelines

MAMBA_130M_HF_REPO_ID = "state-spaces/mamba-130m-hf"
MAMBA_130M_HF_REVISION = hf_repo_lock.revision_for_hf_repo(MAMBA_130M_HF_REPO_ID)

logger = logging.getLogger("max.pipelines")


def test_mamba_130m_hf_generation(capsys: pytest.CaptureFixture[str]) -> None:
    """Test running state-spaces/mamba-130m-hf using the mamba pipeline."""
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                MAMBA_130M_HF_REPO_ID,
                "--prompt",
                "The capital of France is",
                "--trust-remote-code",
                "--devices=cpu",
                "--huggingface-model-revision",
                MAMBA_130M_HF_REVISION,
                "--max-new-tokens=20",
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Expected output from model generation"


def test_mamba_130m_hf_generation_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    """Test running state-spaces/mamba-130m-hf using the mamba pipeline on GPU."""
    assert isinstance(MAMBA_130M_HF_REVISION, str), (
        "MAMBA_130M_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                MAMBA_130M_HF_REPO_ID,
                "--prompt",
                "The capital of France is",
                "--trust-remote-code",
                "--devices=gpu:0",
                "--huggingface-model-revision",
                MAMBA_130M_HF_REVISION,
                "--max-new-tokens=20",
                "--top-k=1",
                "--max-batch-size=1",
                "--max-length=512",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Expected output from model generation"

