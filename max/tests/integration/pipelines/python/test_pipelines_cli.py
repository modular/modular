# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest
from max.entrypoints import pipelines


def test_pipelines_cli__smollm_float32(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
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


def test_pipelines_cli__custom_model(capsys):
    path = os.getenv("PIPELINES_CUSTOM_ARCHITECTURE")
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                "HuggingFaceTB/SmolLM-135M",
                "--weight-path",
                "QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q4_K_M.gguf",
                "--prompt",
                "Why is the sky blue?",
                "--trust-remote-code",
                "--quantization-encoding=q4_k",
                "--device-memory-utilization=0.1",
                f"--custom-architectures={path}",
            ]
        )
    captured = capsys.readouterr()
    assert (
        # Normally q4_k is supported by max engine for llama.
        # We override with a dummy model that does not support q4_k.
        # So this failing shows that the custom model was loaded.
        "'SupportedEncoding.q4_k' not supported" in captured.err
    )
