# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.fn @"__copyinit__{{.*}}(%other: {{.*}}!MemType1{{.*}}read_mem
# CHECK-SAME: %self: !lit.ref<!MemType1, mut {{.*}}> byref_result)
# CHECK-NEXT:   [[M0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:   [[existing_impl:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:   [[loaded_existing_impl:%.*]] = lit.ref.load [[existing_impl]]
# CHECK-NEXT:   lit.ref.store [[loaded_existing_impl]], [[M0]]
# CHECK-NEXT:   [[M1:%.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT:   [[M2:%.*]] = lit.ref.struct.ger %other[dtor]
# CHECK-NEXT:   [[M3:%.*]] = lit.ref.load [[M2]]
# CHECK-NEXT:   lit.ref.store [[M3]], [[M1]]
# CHECK-NEXT:   [[M4:%.*]] = lit.ref.struct.ger %self[_copy]
# CHECK-NEXT:   [[M5:%.*]] = lit.ref.struct.ger %other[_copy]
# CHECK-NEXT:   [[M6:%.*]] = lit.ref.load [[M5]]
# CHECK-NEXT:   lit.ref.store [[M6]], [[M4]]
# CHECK-NEXT:   [[M7:%.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT:   [[M8:%.*]] = lit.ref.struct.ger %other[call]
# CHECK-NEXT:   [[M9:%.*]] = lit.ref.load [[M8]]
# CHECK-NEXT:   lit.ref.store [[M9]], [[M7]]
# CHECK-NEXT:   kgen.param.constant: none
# CHECK-NEXT:   [[W0:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:   [[W1:%.*]] = lit.ref.load [[W0]]
# CHECK-NEXT:   [[W2:%.*]] = lit.ref.struct.ger %self[_copy]
# CHECK-NEXT:   [[W4:%.*]] = lit.ref.load [[W2]]

# Call the copy constructor member with the uninitialized self and the untyped existing impl.
# CHECK-NEXT:  [[W5:%.*]] = lit.call_indirect [[W4]]([[W1]])
# CHECK-NEXT:  [[W3:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:  store [[W5]], [[W3]]
# CHECK-NEXT:  lit.return
# CHECK-NEXT:  lit.end_fn

# CHECK-LABEL: lit.fn @"materialize_escaping_closure

# CHECK: lit.fn @"fn{{.*}}_copyinit_`_CI_{{.*}}(%other: !kgen.pointer<none>, |)

# Allocate memory on the heap for impl and copy existing contents into it.
# CHECK-NEXT:  %[[SIZEOF:.*]] = kgen.param.constant = <get_sizeof(
# CHECK-NEXT:  %[[ALIGNOF:.*]] = kgen.param.constant = <get_alignof(
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %[[ALIGNOF]], %[[SIZEOF]]
# CHECK-NEXT:  %[[V1:.*]] = pop.pointer.bitcast %other
# CHECK-NEXT:  %[[REF0:.*]] = lit.ref.from_pointer %[[V0]]
# CHECK-NEXT:  %[[REF1:.*]] = lit.ref.from_pointer %[[V1]]
# CHECK-NEXT:  %[[REF2:.*]] = lit.ref.immut %[[REF1]]
# CHECK-NEXT:  lit.call {{.*}}__copyinit__{{.*}}(%[[REF2]], %[[REF0]])

# Store the address of the heap allocated memory into the self.
# CHECK-NEXT:  [[V4:%.*]] = pop.pointer.bitcast %[[V0]]
# CHECK-NEXT:  return [[V4]]


@fieldwise_init
struct MemType(Copyable, Movable):
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


fn materialize_escaping_closure(m: MemType):
    fn unique(n: MemType) -> MemType:
        return m + n
