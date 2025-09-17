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
#   Path:   tests/data/py_38/pep_572_remove_parens.py
#
# ===----------------------------------------------------------------------=== #

if foo := 0:
    pass

if foo := 1:
    pass

if y := 5 + 5:
    pass

y = (x := 0)

y += (x := 0)

(y := 5 + 5)

test: int = (test2 := 2)

a, b = (test := (1, 2))

# see also https://github.com/psf/black/issues/2139
assert (foo := 42 - 12)

foo(x=(y := f(x)))


def foo(answer=(p := 42)):
    ...


def foo2(answer: (p := 42) = 5):
    ...


lambda: (x := 1)

a[(x := 12)]
a[:(x := 13)]

# we don't touch expressions in f-strings but if we do one day, don't break 'em
f"{(x:=10)}"


def a():
    return (x := 3)
    await (b := 1)
    yield (a := 2)
    raise (c := 3)


def this_is_so_dumb() -> (please := no):
    pass


# output
if foo := 0:
    pass

if foo := 1:
    pass

if y := 5 + 5:
    pass

y = (x := 0)

y += (x := 0)

(y := 5 + 5)

test: int = (test2 := 2)

a, b = (test := (1, 2))

# see also https://github.com/psf/black/issues/2139
assert (foo := 42 - 12)

foo(x=(y := f(x)))


def foo(answer=(p := 42)):
    ...


def foo2(answer: (p := 42) = 5):
    ...


lambda: (x := 1)

a[(x := 12)]
a[:(x := 13)]

# we don't touch expressions in f-strings but if we do one day, don't break 'em
f"{(x:=10)}"


def a():
    return (x := 3)
    await (b := 1)
    yield (a := 2)
    raise (c := 3)


def this_is_so_dumb() -> (please := no):
    pass
