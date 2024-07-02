# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that just using the function type generates a closure wrapper.
# COM: Check all the generated methods.


@value
struct MemType:
    pass


# CHECK-LABEL: lit.struct.decl @"fn
# CHECK:         lit.struct.field field0 : !kgen.pointer<none>
# CHECK-NEXT:    lit.struct.field dtor : {{.*}}<("self": !kgen.pointer<none>, |) -> !kgen.none>
# CHECK-NEXT:    lit.struct.field copy : {{.*}}<("other": !kgen.pointer<none>, |) -> !kgen.pointer<none>>
# CHECK-NEXT:    lit.struct.field call : {{.*}}<[1](!kgen.pointer<none>, |, ?, "__result__": !lit.ref<!MemType, mut *[0,0]> byref_result) -> !kgen.none>

# CHECK-LABEL:   lit.func @"__del__
# CHECK-NEXT:      [[REF_TO_IMPL:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:      [[OPAQUE_IMPL:%.*]] = lit.ref.load [[REF_TO_IMPL]]
# CHECK-NEXT:      [[DTOR_PTR:%.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT:      [[DTOR:%.*]] = lit.ref.load [[DTOR_PTR]]
# CHECK-NEXT:      lit.call_indirect [[DTOR]]([[OPAQUE_IMPL]])
# CHECK-NEXT:      kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:      lit.ownership.mark_destroyed %self
# CHECK-NEXT:      lit.return %none : !kgen.none
# CHECK-NEXT:      lit.end_func

# CHECK-LABEL:   lit.func @"__copyinit__
# CHECK-NEXT:      [[P0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:      [[existing_impl:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:      [[loaded_existing_impl:%.*]] = lit.ref.load [[existing_impl]]
# CHECK-NEXT:      lit.ref.store [[loaded_existing_impl]], [[P0]]
# CHECK-NEXT:      [[P1:%.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT:      [[P2:%.*]] = lit.ref.struct.ger %other[dtor]
# CHECK-NEXT:      [[P3:%.*]] = lit.ref.load [[P2]]
# CHECK-NEXT:      lit.ref.store [[P3]], [[P1]]
# CHECK-NEXT:      [[P4:%.*]] = lit.ref.struct.ger %self[copy]
# CHECK-NEXT:      [[P5:%.*]] = lit.ref.struct.ger %other[copy]
# CHECK-NEXT:      [[P6:%.*]] = lit.ref.load [[P5]]
# CHECK-NEXT:      lit.ref.store [[P6]], [[P4]]
# CHECK-NEXT:      [[P7:%.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT:      [[P8:%.*]] = lit.ref.struct.ger %other[call]
# CHECK-NEXT:      [[P9:%.*]] = lit.ref.load [[P8]]
# CHECK-NEXT:      lit.ref.store [[P9]], [[P7]]
# CHECK-NEXT:      kgen.param.constant: none
# CHECK-NEXT:      [[EXISTING_IMPL_PTR:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:      [[EXISTING_IMPL:%.*]] = lit.ref.load [[EXISTING_IMPL_PTR]]
# CHECK-NEXT:      [[COPY_PTR:%.*]] = lit.ref.struct.ger %self[copy]
# CHECK-NEXT:      [[COPY:%.*]] = lit.ref.load [[COPY_PTR]]
# CHECK-NEXT:      [[NEW:%.*]] = lit.call_indirect [[COPY]]([[EXISTING_IMPL]])
# CHECK-NEXT:      [[SELF_IMPL_REF:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:      store [[NEW]], [[SELF_IMPL_REF]]

# CHECK-LABEL:  lit.func @"__moveinit__
# CHECK-NEXT:     [[M0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:     [[mov_existing_impl:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:     [[mov_loaded_existing_impl:%.*]] = lit.load.consume [[mov_existing_impl]]
# CHECK-NEXT:     lit.ref.store [[mov_loaded_existing_impl]], [[M0]]
# CHECK-NEXT:     [[M1:%.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT:     [[M2:%.*]] = lit.ref.struct.ger %other[dtor]
# CHECK-NEXT:     [[M3:%.*]] = lit.load.consume [[M2]]
# CHECK-NEXT:     lit.ref.store [[M3]], [[M1]]
# CHECK-NEXT:     [[M4:%.*]] = lit.ref.struct.ger %self[copy]
# CHECK-NEXT:     [[M5:%.*]] = lit.ref.struct.ger %other[copy]
# CHECK-NEXT:     [[M6:%.*]] = lit.load.consume [[M5]]
# CHECK-NEXT:     lit.ref.store [[M6]], [[M4]]
# CHECK-NEXT:     [[M7:%.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT:     [[M8:%.*]] = lit.ref.struct.ger %other[call]
# CHECK-NEXT:     [[M9:%.*]] = lit.load.consume [[M8]]
# CHECK-NEXT:     lit.ref.store [[M9]], [[M7]]
# CHECK-NEXT:     %pointer = kgen.param.constant: pointer<none> = <0>
# CHECK-NEXT:     [[V0:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:     lit.ref.store %pointer, [[V0]]
# CHECK-NEXT:     [[V3:%.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:     lit.ownership.mark_destroyed %other


fn thing(x: fn () escaping -> MemType):
    pass
