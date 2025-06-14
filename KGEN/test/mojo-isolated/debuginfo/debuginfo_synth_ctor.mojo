# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: Synthesized constructors should not emit debuginfo.

# CHECK: lit.struct.decl @MyValueStruct(!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility)
# CHECK-SAME: attributes {sourceName = #MyValueStruct_name}

# CHECK-NOT: #debuginfo.subprogram


@fieldwise_init
struct MyValueStruct(Copyable, Movable, ExplicitlyCopyable):
    var value: __mlir_type.index
