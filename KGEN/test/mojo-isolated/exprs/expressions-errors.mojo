# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


fn takes_pos_or_kw_arg(i: int, j: int):
    pass


fn test_duplicate_kw_arg(x: int):
    takes_pos_or_kw_arg(
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword argument 'j'}}
    )


fn test_pos_after_kw_arg(x: int):
    takes_pos_or_kw_arg(
        j=x,
        x,  # expected-error {{positional argument follows keyword argument}}
    )


fn takes_pos_or_kw_param[i: int, j: int]():
    pass


fn test_duplicate_kw_param[x: int]():
    takes_pos_or_kw_param[
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword parameter 'j'}}
    ]


fn test_pos_after_kw_param[x: int]():
    takes_pos_or_kw_param[
        j=x,
        x,  # expected-error {{positional parameter follows keyword parameter}}
    ]


fn invalid_with():
    # expected-error @below {{use of unknown declaration 'bogus'}}
    with bogus() as foo:
        foo.something()
