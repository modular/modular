# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: Synthesized constructors should not emit debuginfo.

# CHECK: lit.struct.decl @MyValueStruct(!AnyType_Copyable_ImplicitlyDeletable_Movable)
# CHECK-SAME: attributes {sourceName = #MyValueStruct_name}

# The only debug info comes from default trait method for copy.
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "copy($0)"

# We also have Moveinit in Movable and __del__ in ImplicitlyDeletable.
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "__init__(move:$0$)"
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}linkageName = "__del__($0$)"

# CHECK-NOT: #debuginfo.subprogram


@fieldwise_init
struct MyValueStruct(Copyable):
    var value: __mlir_type.index
