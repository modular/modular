# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.entrypoints import pipelines


def test_pipelines_list(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pipelines.main(["list"])
    captured = capsys.readouterr()
    assert "DeepseekV2ForCausalLM" in captured.out
