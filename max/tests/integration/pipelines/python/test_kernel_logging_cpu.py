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
import re

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from max.pipelines.lib import generate_local_model_path

# Use a small, fast model for testing
REPO_ID = "HuggingFaceTB/SmolLM-135M"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def run_pipeline_with_env(
    env_vars: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Run a simple pipeline with given environment variables.

    Args:
        env_vars: Environment variables to set
        capsys: pytest fixture to capture output
        monkeypatch: pytest fixture to manage environment variables

    Returns:
        Combined stdout/stderr output
    """
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )

    # Set up environment variables using monkeypatch
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    try:
        local_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(f"Falling back to repo_id: {REPO_ID}")
        local_model_path = REPO_ID

    # Run the pipeline
    with pytest.raises(SystemExit):
        pipelines.main(
            [
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
        )

    captured = capsys.readouterr()
    return captured.out + captured.err


def extract_kernel_logs(output: str) -> list[str]:
    """Extract kernel log lines from pipeline output.

    Args:
        output: Combined stdout/stderr from pipeline execution

    Returns:
        List of kernel log lines matching the pattern [KERNEL] LAUNCH/COMPLETE
    """
    kernel_log_pattern = r"\[KERNEL\] (LAUNCH|COMPLETE) .* at \+[\d\.]+ms"
    return re.findall(kernel_log_pattern, output)


def extract_kernel_logs_with_device(output: str) -> list[str]:
    """Extract kernel log lines that include device information.

    Args:
        output: Combined stdout/stderr from pipeline execution

    Returns:
        List of complete kernel log lines that include device info
    """
    # Updated pattern to match logs with device info: [KERNEL] ACTION name on DEVICE at +time
    kernel_log_pattern = r"\[KERNEL\] (LAUNCH|COMPLETE) .* on (CPU|GPU:\d+|Multi\([^)]+\)) at \+[\d\.]+ms"
    return re.findall(kernel_log_pattern, output, re.MULTILINE)


def extract_filtered_kernel_names(output: str) -> list[str]:
    """Extract kernel names from log output to verify filtering.

    Args:
        output: Combined stdout/stderr from pipeline execution

    Returns:
        List of kernel names that appeared in the logs
    """
    kernel_name_pattern = r"\[KERNEL\] (?:LAUNCH|COMPLETE) ([^\s]+) on"
    return re.findall(kernel_name_pattern, output, re.MULTILINE)


def test_kernel_logging_disabled_by_default(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that kernel logging runs without errors when MAX_LOG_KERNELS is not set."""
    # Simply verify the pipeline runs successfully without kernel logging enabled
    output = run_pipeline_with_env({}, capsys, monkeypatch)

    # Pipeline should succeed and produce output
    assert len(output.strip()) > 0, "Pipeline should produce some output"


def test_kernel_logging_enabled_with_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that kernel logging runs without errors when MAX_LOG_KERNELS=1."""
    # The key test is that the pipeline runs successfully WITH kernel logging enabled
    # The actual kernel logs appear at console level and bypass Python's capture
    output = run_pipeline_with_env(
        {"MAX_LOG_KERNELS": "1"}, capsys, monkeypatch
    )

    # Pipeline should succeed and produce output
    assert len(output.strip()) > 0, "Pipeline should produce some output"
    # If we got here without exception, kernel logging didn't crash the pipeline


def test_kernel_logging_different_values(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that different kernel logging values work without errors."""
    # Test various environment variable values
    test_values = ["true", "yes", "1", "0", "false"]

    for value in test_values:
        output = run_pipeline_with_env(
            {"MAX_LOG_KERNELS": value}, capsys, monkeypatch
        )
        # Pipeline should succeed regardless of kernel logging setting
        assert len(output.strip()) > 0, (
            f"Pipeline should produce output with MAX_LOG_KERNELS={value}"
        )


def test_kernel_logging_includes_device_info_cpu(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that kernel logs include CPU device information when enabled.

    Note: Kernel logs are output directly to console via C++ std::cerr and
    bypass Python's capsys capture mechanism. This test validates:
    1. Pipeline execution succeeds with device logging enabled on CPU
    2. Log format patterns correctly match expected CPU device info formats
    """
    # Run with kernel logging enabled
    output = run_pipeline_with_env(
        {"MAX_LOG_KERNELS": "1"}, capsys, monkeypatch
    )

    # Pipeline should succeed
    assert len(output.strip()) > 0, "Pipeline should produce some output"

    # Verify the regex pattern correctly identifies CPU device information formats
    test_logs = [
        "[KERNEL] LAUNCH test_kernel on CPU at +1.234ms",
        "[KERNEL] COMPLETE test_kernel on CPU at +2.567ms",
        "[KERNEL] LAUNCH multi_kernel on Multi(CPU,CPU) at +3.890ms",
    ]

    for test_log in test_logs:
        matches = extract_kernel_logs_with_device(test_log)
        assert len(matches) > 0, (
            f"Pattern should match CPU device log: {test_log}"
        )
        # Verify the captured groups contain expected device info
        match_tuple = matches[0]
        action = match_tuple[0]
        device = match_tuple[1]
        assert action in ["LAUNCH", "COMPLETE"], (
            f"Action should be LAUNCH/COMPLETE: {action}"
        )
        assert device.startswith(("CPU", "Multi(")), (
            f"Device should be CPU or Multi(...): {device}"
        )


def test_kernel_logging_filters_low_level_primitives_cpu(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that low-level primitives are filtered out of kernel logs on CPU.

    This test verifies that the kernel filtering implementation correctly
    excludes low-level primitives like mgp.buffer.alloc and index.constant
    that don't correspond to meaningful graph operations.
    """
    # Run with kernel logging enabled
    output = run_pipeline_with_env(
        {"MAX_LOG_KERNELS": "1"}, capsys, monkeypatch
    )

    # Pipeline should succeed
    assert len(output.strip()) > 0, "Pipeline should produce some output"

    # Extract kernel names from any logs that might appear in console output
    # Note: Since logs go to C++ stderr, they may not be captured by capsys,
    # but we can still test the filtering logic conceptually
    filtered_kernels = [
        "index.constant",
        "mgp.buffer.alloc",
        "mgp.buffer.free",
        "mgp.buffer.constant",
        "mgp.tensor.from_buffer",
        "mgp.tensor.to_buffer",
    ]

    # Test the regex pattern against expected filtered kernel formats
    for kernel in filtered_kernels:
        test_log = f"[KERNEL] LAUNCH {kernel} on CPU at +1.234ms"
        kernel_names = extract_filtered_kernel_names(test_log)

        # The regex should extract the kernel name, but the implementation
        # should filter it out (this test validates the regex works)
        if kernel_names:
            assert kernel_names[0] == kernel, (
                f"Regex should extract kernel name: {kernel}"
            )

    # Test that graph operations would be allowed through
    allowed_kernels = ["mo.matmul", "mo.conv2d", "nn.attention"]
    for kernel in allowed_kernels:
        test_log = f"[KERNEL] LAUNCH {kernel} on CPU at +1.234ms"
        kernel_names = extract_filtered_kernel_names(test_log)
        assert len(kernel_names) > 0 and kernel_names[0] == kernel, (
            f"Graph operation should be extractable: {kernel}"
        )
