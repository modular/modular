# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from max.pipelines.lib import generate_local_model_path

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


@pytest.mark.skip("AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16(capsys):
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
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--huggingface-model-revision",
                REVISION,
            ]
        )
    captured = capsys.readouterr()
    # With an Instruct model, and as generate does not have prompt
    # templating for chat, this will just complete the question before resolving.
    assert "?" in captured.out


@pytest.mark.skip("AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16_with_structured_output_enabled(capsys):
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
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--huggingface-model-revision",
                REVISION,
                # We are enabling structured output here, to ensure that
                # enabling structured output server wide, without a JSON
                # schema provided does not change the outputs of the base
                # chat experience.
                "--enable-structured-output",
            ]
        )
    captured = capsys.readouterr()
    # With an Instruct model, and as generate does not have prompt
    # templating for chat, this will just complete the question before resolving.
    assert "?" in captured.out


@pytest.mark.skip("AITLIB-341: Some I/O operation failure")
def test_pipelines_cli__speculative_decoding(capsys):
    try:
        draft_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        draft_model_path = REPO_ID

    local_repo_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    local_revision = hf_repo_lock.revision_for_hf_repo(local_repo_id)
    try:
        local_model_path = generate_local_model_path(
            local_repo_id, local_revision
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {local_repo_id} as config to PipelineConfig"
        )
        local_model_path = local_repo_id

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--draft-model-path",
                draft_model_path,
                "--prompt",
                "hey, hows it going?",
                "--device-memory-utilization=0.3",
                "--max-num-steps=5",
                "--no-enable-chunked-prefill",
            ]
        )

    captured = capsys.readouterr()

    assert "I was just thinking about our last chat" in captured.out
