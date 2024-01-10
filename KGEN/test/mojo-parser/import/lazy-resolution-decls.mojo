# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Run parsing twice to ensure the cache is populated.
# RUN: kgen-translate -mojo-disable-builtins -import-mojo -I=%S %s -o /dev/null
# RUN: kgen-translate -mojo-disable-builtins -import-mojo -I=%S %s | FileCheck %s

from imported_cached_module import StringLiteralAlias, global_variable, Trait


# CHECK-LABEL: lit.func @"assign_from()"
fn assign_from():
    # CHECK: string = <"foobar">
    let foo = StringLiteralAlias
    # CHECK: lit.globalvar.ref {{.*}}@global_variable
    let bar = global_variable


# CHECK-LABEL: lit.struct.decl @Struct(trait<@"$imported_cached_module"::@Trait>, trait<{{.*}}@AnyType>[{{.*}}])
struct Struct(Trait):
    pass
