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
#   Path:   src/black/rusty.py
#
# ===----------------------------------------------------------------------=== #

"""An error-handling model influenced by that used by the Rust programming language

See https://doc.rust-lang.org/book/ch09-00-error-handling.html.
"""
from typing import Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class Ok(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value = value

    def ok(self) -> T:
        return self._value


class Err(Generic[E]):
    def __init__(self, e: E) -> None:
        self._e = e

    def err(self) -> E:
        return self._e


Result = Union[Ok[T], Err[E]]
