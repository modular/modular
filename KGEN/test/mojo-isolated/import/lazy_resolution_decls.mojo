# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Run parsing twice to ensure the cache is populated.
# RUN: %parse-mojo-isolated -I=%S %s -o /dev/null

# CHECK: !Trait = !lit.trait<@imported_cached_module::@Trait>

from imported_cached_module import (
    StringLiteralAlias,
    __global_variable,
    Trait,
    FuncRefField,
)


# CHECK-LABEL: lit.fn @"assign_from()"
fn assign_from():
    # CHECK: string = <"foobar">
    var foo = StringLiteralAlias

    # CHECK: lit.globalvar.ref {{.*}}@__global_variable
    var bar = __global_variable


# CHECK-LABEL: lit.struct.decl @Struct(!Trait, !AnyType[!Trait])
struct Struct(Trait):
    pass


# CHECK-LABEL: lit.file_module @imported_cached_module
# CHECK: lit.struct.field func_ref : {{.*}}@FuncRefField::@"foo()"
fn pull_symbol(x: FuncRefField):
    pass
