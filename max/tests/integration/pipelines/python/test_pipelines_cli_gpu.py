# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from max.pipelines import PIPELINE_REGISTRY


def test_pipelines_cli__smollm_bfloat16(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--prompt",
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--huggingface-revision",
                hf_repo_lock.revision_for_hf_repo(
                    "HuggingFaceTB/SmolLM2-135M-Instruct"
                ),
            ]
        )
    captured = capsys.readouterr()
    # With an Instruct model, and as generate does not have prompt
    # templating for chat, this will just complete the question before resolving.
    assert "?" in captured.out


def test_pipelines_cli__smollm_bfloat16_with_structured_output_enabled(capsys):
    PIPELINE_REGISTRY.reset()
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--prompt",
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--huggingface-revision",
                hf_repo_lock.revision_for_hf_repo(
                    "HuggingFaceTB/SmolLM2-135M-Instruct"
                ),
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
