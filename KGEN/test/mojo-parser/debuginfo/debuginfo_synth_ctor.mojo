# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: Synthesized constructors should not emit debuginfo.

# CHECK: lit.struct.decl @MyValueStruct(!AnyType_Copyable_ImplicitlyDestructible_Movable)
# CHECK-SAME: attributes {sourceName = #MyValueStruct_name}

# The only debug info comes from default trait method for copy.
# CHECK: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, sourceName = #{{.*}}, linkageName = "copy($0)"
# CHECK-NOT: #debuginfo.subprogram


@fieldwise_init
struct MyValueStruct(Copyable):
    var value: __mlir_type.index
