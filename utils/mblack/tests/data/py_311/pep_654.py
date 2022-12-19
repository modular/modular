# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/py_311/pep_654.py
#
# ===----------------------------------------------------------------------=== #

try:
    raise OSError("blah")
except* ExceptionGroup as e:
    pass


try:
    async with trio.open_nursery() as nursery:
        # Make two concurrent calls to child()
        nursery.start_soon(child)
        nursery.start_soon(child)
except* ValueError:
    pass

try:
    try:
        raise ValueError(42)
    except:
        try:
            raise TypeError(int)
        except* Exception:
            pass
        1 / 0
except Exception as e:
    exc = e

try:
    try:
        raise FalsyEG("eg", [TypeError(1), ValueError(2)])
    except* TypeError as e:
        tes = e
        raise
    except* ValueError as e:
        ves = e
        pass
except Exception as e:
    exc = e

try:
    try:
        raise orig
    except* (TypeError, ValueError) as e:
        raise SyntaxError(3) from e
except BaseException as e:
    exc = e

try:
    try:
        raise orig
    except* OSError as e:
        raise TypeError(3) from e
except ExceptionGroup as e:
    exc = e
