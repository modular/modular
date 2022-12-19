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
#   Path:   tests/data/preview/return_annotation_brackets.py
#
# ===----------------------------------------------------------------------=== #

# Control
def double(a: int) -> int:
    return 2 * a


# Remove the brackets
def double(a: int) -> (int):
    return 2 * a


# Some newline variations
def double(a: int) -> (int):
    return 2 * a


def double(a: int) -> (int):
    return 2 * a


def double(a: int) -> (int):
    return 2 * a


# Don't lose the comments
def double(a: int) -> (int):  # Hello
    return 2 * a


def double(a: int) -> (int):  # Hello
    return 2 * a


# Really long annotations
def foo() -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


def foo() -> intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds:
    return 2


def foo() -> intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds | intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds:
    return 2


def foo(
    a: int,
    b: int,
    c: int,
) -> intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds:
    return 2


def foo(
    a: int,
    b: int,
    c: int,
) -> intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds | intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds:
    return 2


# Split args but no need to split return
def foo(
    a: int,
    b: int,
    c: int,
) -> int:
    return 2


# Deeply nested brackets
# with *interesting* spacing
def double(a: int) -> (((((int))))):
    return 2 * a


def double(a: int) -> (((((int))))):
    return 2 * a


def foo() -> (
    (
        (
            intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
        )
    )
):
    return 2


# Return type with commas
def foo() -> (tuple[int, int, int]):
    return 2


def foo() -> tuple[
    loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
    loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
    loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
]:
    return 2


# Magic trailing comma example
def foo() -> tuple[
    int,
    int,
    int,
]:
    return 2


# Long string example
def frobnicate() -> "ThisIsTrulyUnreasonablyExtremelyLongClassName | list[ThisIsTrulyUnreasonablyExtremelyLongClassName]":
    pass


# output
# Control
def double(a: int) -> int:
    return 2 * a


# Remove the brackets
def double(a: int) -> int:
    return 2 * a


# Some newline variations
def double(a: int) -> int:
    return 2 * a


def double(a: int) -> int:
    return 2 * a


def double(a: int) -> int:
    return 2 * a


# Don't lose the comments
def double(a: int) -> int:  # Hello
    return 2 * a


def double(a: int) -> int:  # Hello
    return 2 * a


# Really long annotations
def foo() -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


def foo() -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


def foo() -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
    | intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


def foo(
    a: int,
    b: int,
    c: int,
) -> intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds:
    return 2


def foo(
    a: int,
    b: int,
    c: int,
) -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
    | intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


# Split args but no need to split return
def foo(
    a: int,
    b: int,
    c: int,
) -> int:
    return 2


# Deeply nested brackets
# with *interesting* spacing
def double(a: int) -> int:
    return 2 * a


def double(a: int) -> int:
    return 2 * a


def foo() -> (
    intsdfsafafafdfdsasdfsfsdfasdfafdsafdfdsfasdskdsdsfdsafdsafsdfdasfffsfdsfdsafafhdskfhdsfjdslkfdlfsdkjhsdfjkdshfkljds
):
    return 2


# Return type with commas
def foo() -> tuple[int, int, int]:
    return 2


def foo() -> (
    tuple[
        loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
        loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
        loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
    ]
):
    return 2


# Magic trailing comma example
def foo() -> (
    tuple[
        int,
        int,
        int,
    ]
):
    return 2


# Long string example
def frobnicate() -> (
    "ThisIsTrulyUnreasonablyExtremelyLongClassName |"
    " list[ThisIsTrulyUnreasonablyExtremelyLongClassName]"
):
    pass
