# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s

# Test that 'comptime' keyword works as a synonym for 'alias'

# Simple comptime declarations
comptime x = 5
comptime y: Int = 10
comptime z = x + y

# Parametric comptime
comptime MyInt[T: AnyType] = T
comptime Add[a: Int, b: Int] = a + b

# In struct
struct MyStruct:
    comptime SIZE = 100
    comptime Type = Int

    fn use_comptime(self) -> Int:
        return Self.SIZE

# In trait
trait MyTrait:
    comptime AssociatedType: AnyType

# Mixing alias and comptime in same file (both should work)
comptime old_style = 42
comptime new_style = 42
