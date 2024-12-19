# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pipelines
import pytest


def test_pipelines_list(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(["list"])
    captured = capsys.readouterr()
    assert "MPTForCausalLM" in captured.out
