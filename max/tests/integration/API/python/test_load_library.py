# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def test_loads_anything() -> None:
    import max._core

    assert max._core.__version__
