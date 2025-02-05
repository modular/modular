# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import max.pipelines.max_pipelines as pipelines
import pytest


def test_pipelines_cli__smollm_float32(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--huggingface-repo-id",
                "HuggingFaceTB/SmolLM-135M",
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=float32",
                "--device-memory-utilization=0.1",
            ]
        )
    captured = capsys.readouterr()
    assert (
        "The sky is blue because of the scattering of light by the Earth’s"
        in captured.out
    )
