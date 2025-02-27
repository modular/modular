# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import hf_repo_lock
import pytest
from max.entrypoints import pipelines


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
