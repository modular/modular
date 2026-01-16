# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test @align decorator for specifying minimum struct alignment.

# CHECK-LABEL: lit.struct.decl @AlignedStruct
# CHECK-SAME: minAlignment = 64 : i64
@align(64)
struct AlignedStruct:
    var x: Int


# CHECK-LABEL: lit.struct.decl @CacheLineAligned
# CHECK-SAME: minAlignment = 128 : i64
@align(128)
struct CacheLineAligned:
    var data: Int


# Test that @align works with other decorators
# CHECK-LABEL: lit.struct.decl @AlignedRegisterPassable
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: minAlignment = 32 : i64
@align(32)
@register_passable("trivial")
struct AlignedRegisterPassable:
    var value: __mlir_type.index


# Test minimum alignment (power of 2) - @align(1) is valid for parametric alignment fallback
# CHECK-LABEL: lit.struct.decl @MinAlign
# CHECK-SAME: minAlignment = 1 : i64
@align(1)
struct MinAlign:
    var x: Int


# CHECK-LABEL: lit.struct.decl @Align2
# CHECK-SAME: minAlignment = 2 : i64
@align(2)
struct Align2:
    var x: Int


# CHECK-LABEL: lit.struct.decl @Align4
# CHECK-SAME: minAlignment = 4 : i64
@align(4)
struct Align4:
    var x: Int


# CHECK-LABEL: lit.struct.decl @LargeAlign
# CHECK-SAME: minAlignment = 4096 : i64
@align(4096)
struct LargeAlign:
    var x: Int


# Test @align on empty struct (no fields)
# CHECK-LABEL: lit.struct.decl @AlignedEmpty
# CHECK-SAME: minAlignment = 64 : i64
@align(64)
struct AlignedEmpty:
    pass


# Test hex literal for alignment value (0x40 == 64)
# CHECK-LABEL: lit.struct.decl @HexAligned
# CHECK-SAME: minAlignment = 64 : i64
@align(0x40)
struct HexAligned:
    var x: Int


# Test decorator order: @register_passable before @align
# CHECK-LABEL: lit.struct.decl @ReversedDecoratorOrder
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: minAlignment = 32 : i64
@register_passable("trivial")
@align(32)
struct ReversedDecoratorOrder:
    var value: __mlir_type.index
