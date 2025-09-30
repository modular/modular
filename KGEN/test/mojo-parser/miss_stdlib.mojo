# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen-translate -import-mojo -mojo-search-paths=nope %s 2>&1 | FileCheck %s
# 

# CHECK: unable to locate module 'stdlib'
# CHECK: fn baz(ignore: Bool):
# CHECK: ^ 
# CHECK: 'stdlib' is required for all normal mojo compiles.
# CHECK: If you see this either:
fn baz(ignore: Bool):
  pass
