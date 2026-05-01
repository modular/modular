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


struct RefGenericAddressSpace(Movable):
    var value: Int

    # CHECK-LABEL: lit.fn @"use_ref
    # CHECK-SAME: AddressSpace
    # CHECK-SAME: %value: !lit.ref<!RefGenericAddressSpace, mut {{.*}}, #lit.struct.extract<
    @staticmethod
    fn use_ref[
        origin: Origin[mut=True], address_space: AddressSpace
    ](ref[origin, address_space] value: Self):
        pass

    # CHECK-LABEL: lit.fn @"use_any_ref
    # CHECK-SAME: value.address_space
    # CHECK-SAME: %value: !lit.ref<!RefGenericAddressSpace, mut {{.*}}, {{.*}}value.address_space{{.*}}>
    @staticmethod
    fn use_any_ref[origin: Origin[mut=True]](
        ref[origin, _] value: Self
    ):
        pass
