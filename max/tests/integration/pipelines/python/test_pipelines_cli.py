# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pipelines
import pytest


def test_foo(capsys):
    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--huggingface-repo-id",
                "modularai/replit-code-1.5",
                "--prompt",
                'def hello():\n print("hello world")\n',
                "--trust-remote-code",
            ]
        )
    captured = capsys.readouterr()
    assert 'if __name__ == "__main__"' in captured.out
