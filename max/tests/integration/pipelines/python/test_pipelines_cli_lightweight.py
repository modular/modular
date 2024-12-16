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
    assert "float32: modularai/replit-code-1.5" in captured.out
