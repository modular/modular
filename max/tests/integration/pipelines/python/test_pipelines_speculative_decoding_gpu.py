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


def test_pipelines_speculative_decoding_gpu(capsys) -> None:  # noqa: ANN001
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    # Use HuggingFace repo ID directly to ensure we have access to all weight formats
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--draft-model-path",
                local_model_path,
                "--model-path",
                local_model_path,
                "--quantization-encoding=float32",
                "--devices=gpu",
                "--cache-strategy=paged",
                "--kv-cache-page-size=128",
                "--max-batch-size=4",
                "--max-num-steps=10",
                "--max-length=512",
                "--device-memory-utilization=0.3",
                '--prompt="The meaning of life is"',
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0
