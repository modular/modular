# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct Parametric[a: int]:
    pass

fn takes_var_params[*a: int]():
    pass


fn test_multiple_unbound_pack():
    # expected-error @+1 {{unbound pack must be at the end of the parameter list}}
    alias t = Parametric[*_, `1`, *_]

# expected-note @+1 {{declared here}}
struct VarParamStruct[*args: Int]:
    pass


fn test_unbound_pack_with_variadic():
    # expected-error @+1 {{unbound pack syntax cannot be used where variadic parameters are expected}}
    VarParamStruct[*_]
    # expected-error @+1 {{cannot read from discard pattern '_'}}
    takes_var_params[*_]


fn test_unpack_non_literal[*a: int]():
    # expected-error @+1 {{unsupported unpack operation}}
    Parametric[*a]
    # expected-error @+1 {{unsupported unpack operation}}
    takes_var_params[*a]


fn test_unbound_pack_arg():
    # expected-error @+1 {{unbound packs not supported yet in runtime arguments}}
    test_unbound_pack_arg(*_)
