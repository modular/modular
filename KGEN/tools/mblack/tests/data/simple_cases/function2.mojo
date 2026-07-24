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
#   Path:   tests/data/simple_cases/function2.py
#
# ===----------------------------------------------------------------------=== #


def f(
    a,
    **kwargs,
) -> A:
    with cache_dir():
        if something:
            result = CliRunner().invoke(
                mblack.main, [str(src1), str(src2), "--diff", "--check"]
            )
    limited.append(-limited.pop())  # negate top
    return A(
        very_long_argument_name1=very_long_value_for_the_argument,
        very_long_argument_name2=-very.long.value.for_the_argument,
        **kwargs,
    )


def g():
    "Docstring."

    def inner():
        pass

    print("Inner defs should breathe a little.")


def h():
    def inner():
        pass

    print("Inner defs should breathe a little.")


if os.name == "posix":
    import termios

    def i_should_be_followed_by_only_one_newline():
        pass

elif os.name == "nt":
    try:
        import msvcrt

        def i_should_be_followed_by_only_one_newline():
            pass

    except ImportError:

        def i_should_be_followed_by_only_one_newline():
            pass

elif False:

    class IHopeYouAreHavingALovelyDay:
        def __call__(self):
            print("i_should_be_followed_by_only_one_newline")

else:

    def foo():
        pass


with hmm_but_this_should_get_two_preceding_newlines():
    pass

# output


def f(
    a,
    **kwargs,
) -> A:
    with cache_dir():
        if something:
            result = CliRunner().invoke(
                mblack.main, [str(src1), str(src2), "--diff", "--check"]
            )
    limited.append(-limited.pop())  # negate top
    return A(
        very_long_argument_name1=very_long_value_for_the_argument,
        very_long_argument_name2=-very.long.value.for_the_argument,
        **kwargs,
    )


def g():
    "Docstring."

    def inner():
        pass

    print("Inner defs should breathe a little.")


def h():
    def inner():
        pass

    print("Inner defs should breathe a little.")


if os.name == "posix":
    import termios

    def i_should_be_followed_by_only_one_newline():
        pass

elif os.name == "nt":
    try:
        import msvcrt

        def i_should_be_followed_by_only_one_newline():
            pass

    except ImportError:

        def i_should_be_followed_by_only_one_newline():
            pass

elif False:

    class IHopeYouAreHavingALovelyDay:
        def __call__(self):
            print("i_should_be_followed_by_only_one_newline")

else:

    def foo():
        pass


with hmm_but_this_should_get_two_preceding_newlines():
    pass
