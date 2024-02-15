# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that a nested function with an effect can form a closure.

# CHECK: lit.struct.decl @"`_CI_{{.*}}
# CHECK: lit.func @"__call__{{.*}}) throws|ownedresult -> !kgen.variant<!Error, !Int>

# CHECK: lit.struct.decl @"fn{{.*}}
# CHECK: lit.func @"__call__{{.*}}) throws|ownedresult -> !kgen.variant<!Error, !Int>

# CHECK: lit.func @"fn{{.*}}_call_`_CI_{{.*}}) throws|ownedresult -> !kgen.variant<!Error, !Int>

fn makes_escaping_closure(m: Int):
   fn two_effects(n: Int) escaping raises -> Int:
      return n + m
