# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %s | FileCheck %s

# Test that parametric aliases include proper type information for all parameters AND result types

# Basic parametric alias with AnyType constraint
# CHECK-DAG: "name": "BasicAlias"
# CHECK-DAG: "type": "AnyType"
comptime BasicAlias[T: AnyType] = T

# Parametric alias with Int parameter
# CHECK-DAG: "name": "IntAlias"
# CHECK-DAG: "type": "AnyStruct[Int]"
comptime IntAlias[size: Int] = Int

# Multiple parameters with different types
# CHECK-DAG: "name": "MultiAlias"
# CHECK-DAG: "type": "AnyType"
comptime MultiAlias[T: AnyType, U: AnyType, count: Int] = T


# Parametric alias with trait constraints
trait TestTrait:
    pass


# CHECK-DAG: "name": "TraitAlias"
# CHECK-DAG: "type": "TestTrait"
comptime TraitAlias[T: TestTrait] = T

# Parametric alias with default values - should still show type info
# CHECK-DAG: "name": "DefaultAlias"
# CHECK-DAG: "type": "AnyType"
comptime DefaultAlias[T: AnyType = Int, value: Int = 42] = T

# Test parametric alias with explicit type annotation (MOTO-1165)
# CHECK-DAG: "name": "TypedAlias"
# CHECK-DAG: "type": "Int"
# CHECK-DAG: "value": "(x + 1)"
comptime TypedAlias[x: Int]: Int = x + 1

# Test non-parametric alias with type annotation
# CHECK-DAG: "name": "SimpleTypedAlias"
# CHECK-DAG: "type": "Int"
# CHECK-DAG: "value": "42"
comptime SimpleTypedAlias: Int = 42
