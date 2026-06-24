# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test: a package whose module re-exports a submodule under an alias
# (`from . import _impl as impl`) must survive being precompiled and reloaded by
# a consumer. Resolving that from-import binds a gated ImportOp over the
# submodule; the now-superseded `unresolved_import` placeholder must be dropped
# before serialization, otherwise reloading the package lists both under the
# alias and fails with "invalid redefinition of 'impl'".

# RUN: mkdir -p %t.dir
# RUN: mojo precompile %S/inputs/submodule_alias_pkg -o %t.dir/submodule_alias_pkg.mojoc
# RUN: mojo run -I %t.dir %s | FileCheck %s

from submodule_alias_pkg.api import api_value


def main():
    # CHECK: 7
    print(api_value())
