# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A submodule of deep_package that is deliberately NOT re-exported by
# deep_package/__init__, used to test that it stays gated even when reached
# through a re-export chain.


def hidden_fn():
    pass
