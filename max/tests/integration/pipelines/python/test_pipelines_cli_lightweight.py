# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.entrypoints import pipelines


def test_pipelines_list(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(["list"])
    captured = capsys.readouterr()
    assert "MPTForCausalLM" in captured.out
