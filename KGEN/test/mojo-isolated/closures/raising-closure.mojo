# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that a nested function with an effect can form a closure.

# CHECK: lit.struct.decl @"`_CI_{{.*}}
# CHECK: lit.func @"__call__{{.*}}, |, %n: !Int, ?, %__error__{{.*}}, %__result__{{.*}}) throws -> i1
# CHECK: lit.raise

# CHECK: lit.struct.decl @"fn{{.*}}
# CHECK: lit.func @"__call__{{.*}}, |, %n: !Int, ?, %__error__{{.*}}, %__result__{{.*}}) throws -> i1
# CHECK: [[IS_ERR:%.*]] = lit.call_indirect {{.*}}%__error__, %__result__)

# CHECK: lit.func @"fn{{.*}}_call_`_CI_{{.*}}) throws -> i1


fn makes_escaping_closure(m: Int) raises:
    fn two_effects(n: Int) escaping raises -> Int:
        raise Error {}

    _ = two_effects(m)
