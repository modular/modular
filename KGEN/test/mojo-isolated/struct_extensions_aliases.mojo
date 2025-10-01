# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -split-input-file %s | FileCheck %s

alias int = __mlir_type.index
alias `42` = __mlir_attr.`42 : index`

struct Spaceship:
    var fuel: int

__extension Spaceship:
    alias MaxSpeed: int = `42`

# CHECK-LABEL: lit.fn @"test_function
fn test_function():
    # CHECK: lit.alias.decl *"MySpeed`"
    alias MySpeed = Spaceship.MaxSpeed
    # CHECK: lit.var.decl "speed"
    var speed: int = MySpeed
