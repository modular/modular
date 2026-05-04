# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

struct MemType(Movable):
    var value: Int

    # Has an explicit address space and origin.

    # CHECK-LABEL: lit.fn @"__init__[::AddressSpace,LITMutOrigin]()"
    # CHECK-SAME: <addr_space: !AddressSpace, ?, *"self_is_origin`2x1": origin<1>>
    # CHECK-SAME: %self: !lit.ref<!MemType, mut {{.*}}, #lit.struct.extract<
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

    # This has a parametric origin like 'ref', but not a parametric addr space.
    # CHECK-LABEL: lit.fn @"__init__[LITMutOrigin](::Int)
    # CHECK-SAME: <?, *"self_is_origin`2x": origin<1>
    # CHECK-SAME: %self: !lit.ref<!MemType, mut *"self_is_origin`2x"> byref_result
    def __init__(a: Int, out[_] self):
        self.value = 0

    # This has a fixed address space, parametric origin.
    # CHECK-LABEL: lit.fn @"__init__[LITMutOrigin](::Int,::Int)
    # CHECK-SAME: %self: !lit.ref<!MemType, mut *"self_is_origin`2x1", sugar_preserved({{.*}}, 1)> byref_result)
    def __init__(a: Int, b: Int, out[AddressSpace.GLOBAL] self):
        self.value = 0


# CHECK-LABEL: lit.struct.decl @RPType
struct RPType(RegisterPassable):
    var value: Int

    # CHECK-LABEL: lit.fn @"__init__[::AddressSpace,LITMutOrigin]()"
    # CHECK-SAME: %self: !lit.ref<!RPType, mut {{.*}}addr_space{{.*}}> byref_result
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

# CHECK-LABEL: lit.fn @"use_out_address_space
def use_out_address_space[addr_space: AddressSpace, o: Origin[mut=True]](
    ref[o] mem1: MemType,
    ref[o, addr_space] mem2: MemType,
    ref[o, AddressSpace.GLOBAL] mem3: MemType,
    ref[o, addr_space] rp: RPType):

   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace {_value: !Int = {0}}, :origin<1> *"o._mlir_origin`">(%mem1)
   mem1 = MemType()

   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace addr_space, :origin<1> *"o._mlir_origin`">(%mem2)
   mem2 = MemType()
   # CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:!AddressSpace {_value: !Int = {_mlir_value = sugar_preserved(#lit.struct.extract<:!Int #lit.struct.extract<:!AddressSpace #kgen.type<!AddressSpace>, "_value">, "_mlir_value">, 1)}}, :origin<1> *"o._mlir_origin`">(%mem3)
   mem3 = MemType()

   # CHECK: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:origin<1> *"o._mlir_origin`">({{.*}}, %mem1)
   mem1 = MemType(0)

   # CHECK: lit.call {{.*}}MemType::@"__init__
   # CHECK-SAME: <:origin<1> *"o._mlir_origin`">({{.*}}, {{.*}}, %mem3)
   mem3 = MemType(0, 0)

   # FIXME: Doesn't work yet.
   #var loc = MemType()

   # CHECK-NEXT: lit.call {{.*}}RPType::@"__init__
   # CHECK-SAME: <:!AddressSpace addr_space, :origin<1> *"o._mlir_origin`">(%rp)
   rp = RPType()
