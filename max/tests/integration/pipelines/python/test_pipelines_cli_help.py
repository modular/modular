# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.entrypoints import pipelines


def test_pipelines_generate_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pipelines.main(["generate", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "generate [OPTIONS]" in captured.out
