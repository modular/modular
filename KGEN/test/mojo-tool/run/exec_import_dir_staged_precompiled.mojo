# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Staged incremental precompilation: packages are precompiled one at a time
# into a plain (non-package) staging directory, and later packages resolve
# their dependencies from the `.mojoc`s already staged there. The second
# `precompile` below is the interesting step: `consumer` imports
# `tmp.helper`, which only exists as `tmp/helper.mojoc` inside the plain
# directory `tmp`.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/stage/tmp
# RUN: mojo precompile %S/inputs/dir_precompiled/helper -o %t.dir/stage/tmp/helper.mojoc
# RUN: mojo precompile -I %t.dir/stage %S/inputs/dir_precompiled/consumer -o %t.dir/stage/tmp/consumer.mojoc
# RUN: mojo run -I %t.dir/stage %s | FileCheck %s

from tmp.consumer import foo


def main():
    # CHECK: precompiled
    foo()
