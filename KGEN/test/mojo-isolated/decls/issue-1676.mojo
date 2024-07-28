# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Issue https://github.com/modularml/mojo/issues/1676

# RUN: not %parse-mojo-isolated -verify-diagnostics %s 2>&1 

@register_passable
struct StructWithoutBody:
    pass


@value
@register_passable
# expected-error @below {{'StructWithoutBody' is not copyable because it has no '__copyinit__'}}
struct OkayStruct:
    var begin: StructWithoutBody

