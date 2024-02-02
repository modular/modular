# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s -verify-diagnostics


fn takes_pos_or_kw_arg(i: Int, j: Int):
    pass


fn test_duplicate_kw_arg(x: Int):
    takes_pos_or_kw_arg(
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword argument 'j'}}
    )


fn test_pos_after_kw_arg(x: Int):
    takes_pos_or_kw_arg(
        j=x,
        x,  # expected-error {{positional argument follows keyword argument}}
    )


fn takes_pos_or_kw_param[i: Int, j: Int]():
    pass


fn test_duplicate_kw_param[x: Int]():
    takes_pos_or_kw_param[
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword parameter 'j'}}
    ]


fn test_pos_after_kw_param[x: Int]():
    takes_pos_or_kw_param[
        j=x,
        x,  # expected-error {{positional parameter follows keyword parameter}}
    ]
