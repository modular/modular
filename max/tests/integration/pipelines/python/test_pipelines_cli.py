# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
import os

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from max.pipelines.lib import generate_local_model_path

REPO_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def test_pipelines_cli__smollm_float32(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        local_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        local_model_path = REPO_ID
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=float32",
                "--device-memory-utilization=0.1",
                "--huggingface-model-revision",
                REVISION,
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()
    assert (
        "The sky is blue because of the scattering of light by the Earth’s"  # noqa: RUF001
        in captured.out
    )


def test_pipelines_cli__custom_model() -> None:
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    path = os.getenv("PIPELINES_CUSTOM_ARCHITECTURE")
    try:
        local_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        local_model_path = REPO_ID

    with pytest.raises(
        ValueError, match=".*'SupportedEncoding.q4_k' not supported.*"
    ):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--weight-path",
                "QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf",
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=q4_k",
                "--device-memory-utilization=0.1",
                f"--custom-architectures={path}",
                "--huggingface-model-revision",
                REVISION,
                "--top-k=1",
            ]
        )
