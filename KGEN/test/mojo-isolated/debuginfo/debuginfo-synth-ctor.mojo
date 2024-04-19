# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: Synthesized constructors should not emit debuginfo.

# CHECK: lit.struct.decl @MyValueStruct(!AnyType, !Copyable, !Movable)  attributes {sourceName = #MyValueStruct_name}

# CHECK-NOT: #debuginfo.subprogram


@value
struct MyValueStruct:
    var value: __mlir_type.index
