# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Integration tests for kernel logging functionality on CPU.

These tests verify that the MAX_LOG_KERNELS environment variable properly
enables kernel launch/completion logging during pipeline execution on CPU.
"""

import logging
import os
import re
import subprocess
import sys

import hf_repo_lock
from max.pipelines.lib import generate_local_model_path

# Use a small, fast model for testing
REPO_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def run_pipeline_subprocess(env_vars: dict[str, str]) -> tuple[str, str]:
    """Run pipeline via subprocess to capture stderr output.

    Args:
        env_vars: Environment variables to set

    Returns:
        Tuple of (stdout, stderr) from pipeline execution
    """
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )

    try:
        local_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(f"Falling back to repo_id: {REPO_ID}")
        local_model_path = REPO_ID

    # Build command to run pipeline directly with python
    # Note: We can't use bazelw from inside a bazel test, so we run the pipeline directly
    cmd = [
        sys.executable,
        "-m",
        "max.entrypoints.pipelines",
        "generate",
        "--model-path",
        local_model_path,
        "--prompt",
        "Hi",
        "--max-new-tokens",
        "1",
        "--trust-remote-code",
        "--quantization-encoding=float32",
        "--device-memory-utilization=0.1",
        "--huggingface-model-revision",
        REVISION,
    ]

    # Prepare environment with our variables
    env = os.environ.copy()
    env.update(env_vars)

    # Run the command and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,  # 5 minute timeout
    )

    # The command might exit with non-zero due to SystemExit in pipeline
    # We still want to capture the output
    return result.stdout, result.stderr


def test_kernel_logging_disabled_by_default() -> None:
    """Test that kernel logging runs without errors when MAX_LOG_KERNELS is not set."""
    # Simply verify the pipeline runs successfully without kernel logging enabled
    stdout, stderr = run_pipeline_subprocess({})

    # Pipeline should succeed and produce output
    assert len(stdout.strip()) > 0, "Pipeline should produce some output"

    # Should not contain any [KERNEL] logs
    kernel_logs = re.findall(r"\[KERNEL\]", stderr)
    assert len(kernel_logs) == 0, "Should not have kernel logs by default"


def test_kernel_logging_different_values() -> None:
    """Test that different kernel logging values work without errors."""
    # Test various environment variable values
    test_values = ["true", "yes", "1"]

    for value in test_values:
        stdout, stderr = run_pipeline_subprocess({"MAX_LOG_KERNELS": value})

        # Pipeline should succeed and produce output
        assert len(stdout.strip()) > 0, "Pipeline should produce some output"

        # Should contain some [KERNEL] logs
        kernel_logs = re.findall(r"\[KERNEL\]", stderr)
        assert len(kernel_logs) != 0, (
            f"Pipeline should produce output with MAX_LOG_KERNELS={value}"
        )


def test_kernel_logging_detailed_output() -> None:
    """Test that actual kernel logs are captured from stderr and have correct format.

    This test validates actual log output by running the pipeline via subprocess
    and capturing stderr, which contains the real kernel logs from llvm::errs().
    """
    # Run pipeline with logging enabled
    stdout, stderr = run_pipeline_subprocess({"MAX_LOG_KERNELS": "1"})

    # Pipeline should succeed
    assert len(stdout.strip()) > 0 or len(stderr.strip()) > 0, (
        "Pipeline should produce some output with logging enabled"
    )

    # Should contain kernel logs when enabled (logs go to stderr)
    kernel_launch_logs = re.findall(r"\[KERNEL\] LAUNCH.*", stderr)
    kernel_complete_logs = re.findall(r"\[KERNEL\] COMPLETE.*", stderr)

    # We expect to see some kernel logs when logging is enabled
    # Note: The exact number may vary based on model execution and caching
    assert len(kernel_launch_logs) > 0 and len(kernel_complete_logs) > 0, (
        "Should have kernel logs when MAX_LOG_KERNELS=1"
    )

    # Validate log format for actual logs we captured
    # Both LAUNCH and COMPLETE logs should have the same format: [KERNEL] ACTION ... on DEVICE at +TIME
    for log in kernel_launch_logs[:5]:  # Check first few logs
        # Should match the expected format: [KERNEL] LAUNCH ... on DEVICE at +TIME
        assert re.match(
            r"\[KERNEL\] LAUNCH .+ on CPU at \+[\d\.]+ms",
            log,
        ), f"LAUNCH log should match expected format: {log}"

    for log in kernel_complete_logs[:5]:  # Check first few logs
        # Should match the expected format: [KERNEL] COMPLETE ... on DEVICE at +TIME
        assert re.match(
            r"\[KERNEL\] COMPLETE .+ on CPU at \+[\d\.]+ms",
            log,
        ), f"COMPLETE log should match expected format: {log}"
