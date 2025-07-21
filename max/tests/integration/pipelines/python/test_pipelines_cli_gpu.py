# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from test_common.graph_utils import is_h100_h200
from test_common.lora_utils import (
    create_multiple_test_lora_adapters,
    create_test_lora_adapter,
)

# Keep original constants for non-LoRA tests
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


logger = logging.getLogger("max.pipelines")


@pytest.mark.skip("AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16(capsys) -> None:  # noqa: ANN001
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
                "--prompt",
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--allow-safetensors-weights-float32-to-bfloat16-cast",
                "--devices=gpu",
                "--huggingface-model-revision",
                REVISION,
            ]
        )
    captured = capsys.readouterr()
    # With an Instruct model, and as generate does not have prompt
    # templating for chat, this will just complete the question before resolving.
    assert "?" in captured.out


@pytest.mark.skip("AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16_with_structured_output_enabled(
    capsys,  # noqa: ANN001
) -> None:
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
                "--prompt",
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--allow-safetensors-weights-float32-to-bfloat16-cast",
                "--devices=gpu",
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


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
def test_pipelines_cli__smollm_with_lora(capsys) -> None:  # noqa: ANN001
    """Test SmolLM2 with LoRA adapter via CLI."""
    from test_common.lora_utils import REPO_ID as LORA_REPO_ID
    from test_common.lora_utils import get_model_revision

    lora_path = create_test_lora_adapter(prefix="cli_test")
    model_path = LORA_REPO_ID
    revision = get_model_revision()

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                model_path,
                "--prompt",
                "What is machine learning?",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--allow-safetensors-weights-float32-to-bfloat16-cast",
                "--devices=gpu",
                "--huggingface-model-revision",
                REVISION,
                "--max-num-loras=2",
                "--max-lora-rank=16",
                f"--lora-paths={lora_path}",
                "--max-new-tokens=50",
            ]
        )
    captured = capsys.readouterr()

    # Verify output contains response about machine learning
    assert len(captured.out) > 0
    # Common words that might appear in ML explanation
    assert any(
        word in captured.out.lower()
        for word in ["learn", "data", "algorithm", "model"]
    )


@pytest.mark.skipif(is_h100_h200(), reason="LoRA tests fail on H100 and H200")
def test_pipelines_cli__smollm_with_multiple_loras(capsys) -> None:  # noqa: ANN001
    """Test SmolLM2 with multiple LoRA adapters via CLI."""
    from test_common.lora_utils import REPO_ID as LORA_REPO_ID
    from test_common.lora_utils import get_model_revision

    # Create multiple LoRA adapters
    lora_adapter_paths = create_multiple_test_lora_adapters(
        num_adapters=2, prefix="cli_multi_test"
    )
    model_path = LORA_REPO_ID
    revision = get_model_revision()

    # Multiple LoRA adapters with names
    lora_paths = [
        f"adapter1={lora_adapter_paths[0]}",
        f"adapter2={lora_adapter_paths[1]}",
    ]

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                model_path,
                "--prompt",
                "Explain quantum computing",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--allow-safetensors-weights-float32-to-bfloat16-cast",
                "--devices=gpu",
                "--huggingface-model-revision",
                REVISION,
                "--max-num-loras=3",
                "--max-lora-rank=16",
                f"--lora-paths={lora_paths[0]}",
                f"--lora-paths={lora_paths[1]}",
                "--max-new-tokens=100",
            ]
        )
    captured = capsys.readouterr()

    # Verify output contains response about quantum computing
    assert len(captured.out) > 0
    assert any(
        word in captured.out.lower()
        for word in ["quantum", "computer", "qubit", "state"]
    )
