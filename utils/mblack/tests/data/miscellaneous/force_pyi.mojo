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
#   Path:   tests/data/miscellaneous/force_pyi.py
#
# ===----------------------------------------------------------------------=== #

from typing import Union


@bird
def zoo():
    ...


class A:
    ...


@bar
class B:
    def BMethod(self) -> None:
        ...

    @overload
    def BMethod(self, arg: List[str]) -> None:
        ...


class C:
    ...


@hmm
class D:
    ...


class E:
    ...


@baz
def foo() -> None:
    ...


class F(A, C):
    ...


def spam() -> None:
    ...


@overload
def spam(arg: str) -> str:
    ...


var: int = 1


def eggs() -> Union[str, int]:
    ...


# output

from typing import Union


@bird
def zoo():
    ...


class A:
    ...


@bar
class B:
    def BMethod(self) -> None:
        ...

    @overload
    def BMethod(self, arg: List[str]) -> None:
        ...


class C:
    ...


@hmm
class D:
    ...


class E:
    ...


@baz
def foo() -> None:
    ...


class F(A, C):
    ...


def spam() -> None:
    ...


@overload
def spam(arg: str) -> str:
    ...


var: int = 1


def eggs() -> Union[str, int]:
    ...
