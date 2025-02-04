# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.pipelines.dataprocessing import max_tokens_to_generate


def test_max_tokens_to_generate():
    assert max_tokens_to_generate(10, 12, -1) == 2
    assert max_tokens_to_generate(3, 25, 8) == 8
    assert max_tokens_to_generate(5, 3, -1) == 0
    assert max_tokens_to_generate(10, 10, -1) == 0
    assert max_tokens_to_generate(10, 10, 3) == 0
