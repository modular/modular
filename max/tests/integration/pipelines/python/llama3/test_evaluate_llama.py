# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests evaluate_llama."""

import re

import evaluate_llama
from click.testing import CliRunner


def test_evaluate_llama():
    runner = CliRunner()
    result = runner.invoke(
        evaluate_llama.main,
        [
            "--model",
            "tinyllama",
            "--encoding",
            "float32",
        ],
    )
    assert result.exit_code == 0
    expected_pattern = r"Goldens for tinyllama float32 written to .*"
    assert re.search(expected_pattern, result.output) is not None
