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


struct SomeType:
    pass


fn mem_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = SomeType


fn reg_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = Int


trait SomeTrait:
    pass


fn trait_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = SomeTrait


fn reg_type_func() -> AnyTrivialRegType:
    # expected-error @below {{dynamic type values not permitted yet}}
    return Int


fn mem_type_func() -> AnyType:
    # expected-error @below {{dynamic type values not permitted yet}}
    return SomeType


fn takes_reg_type(t: AnyTrivialRegType):
    pass


fn test_takes_reg_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(Int)


fn takes_mem_type(t: AnyType):
    pass


fn test_takes_mem_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(SomeType)
