# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

struct InitIndexAddressSpace(Movable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__{{\[index\]\(\)}}
    # CHECK-SAME: <address_space>
    # CHECK-SAME: %self: !lit.ref<!InitIndexAddressSpace, mut {{.*}}, address_space> byref_result
    fn __init__[address_space: __mlir_type.index](out[address_space] self):
        self.value = 0


struct InitGenericOutAddressSpace(Movable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__{{\[::AddressSpace\]\(\)}}
    # CHECK-SAME: AddressSpace
    # CHECK-SAME: %self: !lit.ref<!InitGenericOutAddressSpace, mut {{.*}}, #lit.struct.extract<
    fn __init__[address_space: AddressSpace](out[address_space] self):
        self.value = 0


struct InitAnyOutAddressSpace(Movable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__{{\[index\]\(\)}}
    # CHECK-SAME: self.address_space
    # CHECK-SAME: %self: !lit.ref<!InitAnyOutAddressSpace, mut {{.*}}, {{.*}}self.address_space{{.*}}> byref_result
    fn __init__(out[_] self):
        self.value = 0
