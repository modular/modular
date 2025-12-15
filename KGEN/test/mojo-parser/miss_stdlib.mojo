# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen-translate -import-mojo -mojo-search-paths=nope %s 2>&1 | FileCheck %s
# 

# CHECK: unable to locate module 'std'
# CHECK: fn baz(ignore: Bool):
# CHECK: ^
# CHECK: 'std' is required for all normal mojo compiles.
# CHECK: If you see this either:
fn baz(ignore: Bool):
  pass
