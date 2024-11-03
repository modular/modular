# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest

import pipelines


def test_foo(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(
            ["replit", "--prompt", 'def hello():\n  print("hello world")\n']
        )
    captured = capsys.readouterr()
    assert 'if __name__ == "__main__"' in captured.out
