# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A `.mojoc` inside a plain (non-package) directory is a valid import target:
# a plain directory is just a namespace, so the usual `.mojoc`-over-`.mojo`
# resolution precedence applies, matching top-level `-I` resolution. Inside a
# source package the `.mojoc` remains unresolvable: a package can't
# legitimately nest a precompiled copy of its own submodules.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/only/tmp %t.dir/mixed/tmp %t.dir/pkg/tmp
# RUN: mojo precompile %S/inputs/dir_precompiled/helper -o %t.dir/only/tmp/helper.mojoc

## A `.mojoc`-only child of a plain directory resolves.
# RUN: mojo run -I %t.dir/only %s | FileCheck %s --check-prefix=PRE

## A `.mojoc`/`.mojo` collision in a plain directory: `.mojoc` wins.
# RUN: cp %t.dir/only/tmp/helper.mojoc %t.dir/mixed/tmp/helper.mojoc
# RUN: cp %S/inputs/dir_precompiled/helper_module.mojo %t.dir/mixed/tmp/helper.mojo
# RUN: mojo run -I %t.dir/mixed %s | FileCheck %s --check-prefix=PRE

## The same `.mojoc` inside a source package is not importable.
# RUN: cp %t.dir/only/tmp/helper.mojoc %t.dir/pkg/tmp/helper.mojoc
# RUN: cp %S/inputs/dir_precompiled/empty_init.mojo %t.dir/pkg/tmp/__init__.mojo
# RUN: not mojo run -I %t.dir/pkg %s 2>&1 | FileCheck %s --check-prefix=PKG

# PRE: precompiled
# PKG: error: unable to locate module 'helper'

from tmp.helper import foo


def main():
    foo()
