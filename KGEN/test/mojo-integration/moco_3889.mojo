# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Verify that importing a package where Factory.make() returns an Inner from
# the same package resolves Inner.__init__ correctly.
#
# Root cause: resolveDeclFromBytecode sets decl.resolvedness = body before
# calling bytecodeReader->materialize().  During the intervening package-dep
# resolution, a FuncSymbolAttr for Inner.__init__ could be walked while the
# parent package op had an empty body region.  getSymbolTable() cached the
# empty SymbolTable, and visitedAttrTypes permanently cached the walk failure,
# so Inner.__init__ was never registered in declForFuncSymbol — triggering the
# KGEN verifier error "does not reference a KGEN declaration".
#
# The fix: (1) invalidate the MLIR symbol-table cache after materialization so
# subsequent lookups see the inflated body; (2) do not permanently cache
# WalkResult::interrupt in visitedAttrTypes so transient pre-materialization
# failures are retried once the package is ready.

# RUN: mkdir -p %t.moco-3889
# RUN: mojo precompile %S/inputs/moco_3889_package -o %t.moco-3889/moco_3889_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.moco-3889 %s | FileCheck %s

from moco_3889_package import Factory


def test() -> Int:
    var f = Factory()
    # CHECK: lit.call @moco_3889_package::@factory::@Factory::@"__init__
    var i = f.make()
    # CHECK: lit.call @moco_3889_package::@factory::@Factory::@"make(
    return i.value


# Verify the imported package is present and that Inner::__init__ was resolved.
# Inner::__init__'s FuncSymbolAttr appears inside Factory.make()'s body in the
# package IR; it must be registered in declForFuncSymbol or the verifier fires.
# CHECK: lit.package @moco_3889_package
# CHECK: lit.fn @"make(
# CHECK: @moco_3889_package::@inner::@Inner::@"__init__(
