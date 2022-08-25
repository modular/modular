##===- test_typing.py -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


def test_imports():
    from modular.utils.typing import (  # noqa: F401
        IO,
        BinaryIO,
        Iterable,
        Iterator,
        Match,
        Pattern,
        TextIO,
        Tuple,
    )
