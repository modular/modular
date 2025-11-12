# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json

from click.testing import CliRunner
from max.tests.integration.pipelines.python.smoke_tests import (
    smoke_test_github_matrix,
)


def test_smoke_test_github_matrix_b200_max_ci() -> None:
    runner = CliRunner()
    result = runner.invoke(
        smoke_test_github_matrix.main,
        ["--framework", "max-ci", "--run-on-b200"],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "include" in output
    assert len(output["include"]) > 0
