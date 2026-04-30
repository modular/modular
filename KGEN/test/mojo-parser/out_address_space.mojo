# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

struct InitIndexAddressSpace(Movable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__
    # CHECK-SAME: <address_space>
    # CHECK-SAME: %self: !lit.ref<!InitIndexAddressSpace, mut {{.*}}, address_space> byref_result
    fn __init__[address_space: __mlir_type.index](out[address_space] self):
        self.value = 0
