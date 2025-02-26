# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Unit test to test generate_llm_logits functionality"""

import generate_llm_logits
from click.testing import CliRunner


def test_generate_llm_logits_smollm(tmp_path):
    runner = CliRunner()
    output_file = tmp_path / "output_goldens.json"
    result = runner.invoke(
        generate_llm_logits.main,
        [
            "--framework=max",
            "--pipeline=smollm",
            "--encoding=bfloat16",
            "--device=gpu",
            "--output",
            f"{output_file}",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")
    assert output_file.exists()
