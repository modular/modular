# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Issue https://github.com/modularml/mojo/issues/1676
# Because some notes emitted by the compiler point to a location in a different
# file, we cannot use an expected-note to check for them. So we pipe it into
# FileCheck instead.

# RUN: not %parse-mojo-isolated -verify-diagnostics %s 2>&1 | FileCheck %s

@register_passable
struct StructWithoutBody:  # expected-error {{expected body statements; use 'pass' if none is required}}


@value
@register_passable
# expected-error @below {{'StructWithoutBody' is not copyable because it has no '__copyinit__'}}
# expected-error @below {{struct 'OkayStruct' does not implement all requirements for 'Copyable'}}
struct OkayStruct:
    var begin: StructWithoutBody

# CHECK: required function '__copyinit__' is not implemented
# CHECK: trait 'Copyable' declared here
