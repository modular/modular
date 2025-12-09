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
import os

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from max.pipelines.lib import generate_local_model_path

SMOLLM_HF_REPO_ID = "HuggingFaceTB/SmolLM-135M"
SMOLLM_HF_REVISION = hf_repo_lock.revision_for_hf_repo(SMOLLM_HF_REPO_ID)

logger = logging.getLogger("max.pipelines")


@pytest.fixture
def smollm_135m_local_path() -> str:
    assert isinstance(SMOLLM_HF_REVISION, str), (
        "SMOLLM_HF_REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        model_path = generate_local_model_path(
            SMOLLM_HF_REPO_ID, SMOLLM_HF_REVISION
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {SMOLLM_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = SMOLLM_HF_REPO_ID
    return model_path


def test_pipelines_cli__smollm_float32(
    smollm_135m_local_path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                smollm_135m_local_path,
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=float32",
                "--device-memory-utilization=0.1",
                "--huggingface-model-revision",
                SMOLLM_HF_REVISION,
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()
    assert (
        "The sky is blue because of the scattering of light by the Earth’s"  # noqa: RUF001
        in captured.out
    )


def test_pipelines_cli__custom_model(smollm_135m_local_path: str) -> None:
    path = os.getenv("PIPELINES_CUSTOM_ARCHITECTURE")

    with pytest.raises(
        ValueError, match=r".*'SupportedEncoding.q4_k' not supported.*"
    ):
        pipelines.main(
            [
                "generate",
                "--model-path",
                smollm_135m_local_path,
                "--weight-path",
                "QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf",
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=q4_k",
                "--device-memory-utilization=0.1",
                f"--custom-architectures={path}",
                "--huggingface-model-revision",
                SMOLLM_HF_REVISION,
                "--top-k=1",
            ]
        )


def test_pipelines_cli__model_and_model_path_conflict(
    smollm_135m_local_path: str,
) -> None:
    """Test that specifying both --model and --model-path raises an error."""

    with pytest.raises(
        ValueError, match="model_path and model cannot both be specified"
    ):
        pipelines.main(
            [
                "generate",
                "--model",
                smollm_135m_local_path,
                "--model-path",
                smollm_135m_local_path,
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=float32",
                "--device-memory-utilization=0.1",
                "--huggingface-model-revision",
                SMOLLM_HF_REVISION,
                "--top-k=1",
            ]
        )
