# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

import hf_repo_lock
import pytest
from max.entrypoints import pipelines

# Keep original constants for non-LoRA tests
REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


logger = logging.getLogger("max.pipelines")


def test_pipelines_multi_gpu_smoke(capsys) -> None:  # noqa: ANN001
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    # Use HuggingFace repo ID directly to ensure we have access to all weight formats
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--devices=gpu:0,1,2,3",
                "--max-batch-size=1",
                "--max-new-tokens=32",
                "--max-num-steps=1",
                "--max-length=512",
                "--huggingface-model-revision",
                REVISION,
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_pipelines_multi_gpu_smoke_with_subgraphs(capsys) -> None:  # noqa: ANN001
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    # Use HuggingFace repo ID directly to ensure we have access to all weight formats
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--devices=gpu:0,1,2,3",
                "--max-batch-size=1",
                "--max-new-tokens=32",
                "--max-num-steps=1",
                "--max-length=512",
                "--use-subgraphs",
                "--huggingface-model-revision",
                REVISION,
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0
