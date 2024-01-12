# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# CHECK: lit.struct.field call : !kgen.signature<!lit.signature<[1](!kgen.pointer<none> borrow, |, "n": !lit.ref<!MemType, {{.*}}> borrow_in_mem, "j": !Int borrow) -> !Int>>
# CHECK: lit.func @"__call__{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |, %n: !lit.ref<!MemType, {{.*}}> borrow_in_mem, %j: !Int borrow) -> !Int
# CHECK-NEXT: [[closure_impl_ref0:%.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: [[closure_impl0:%.*]] = lit.ref.load [[closure_impl_ref0]]
# CHECK-NEXT: [[casting_call_ref0:%.*]] = lit.ref.struct.ger %[[SELF]][call]
# CHECK-NEXT: [[casting_call0:%.*]] = lit.ref.load [[casting_call_ref0]]
# CHECK-NEXT: [[result_of_typed_call0:%.*]] = lit.call_signature [[casting_call0]]{{.*}}([[closure_impl0]], %n, %j)
# CHECK-NEXT: lit.return [[result_of_typed_call0]] : !Int
# CHECK-NEXT: lit.end_func
# CHECK-NEXT: }

# CHECK: lit.struct.field call : !kgen.signature<!lit.signature<[2](!lit.ref<mut !MemType, *[0,0]> byref_result, !kgen.pointer<none> borrow, |, "n": !lit.ref<!MemType, {{.*}}> borrow_in_mem) -> !kgen.none>>
# CHECK: lit.func @"__call__{{.*}}(%[[RES:.*]][{{.*}}]: !lit.ref<mut !MemType, {{.*}}> byref_result, %[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |, %n: !lit.ref<!MemType, {{.*}}> borrow_in_mem)
# CHECK-NEXT: [[closure_impl_ref:%.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: [[closure_impl:%.*]] = lit.ref.load [[closure_impl_ref]]
# CHECK-NEXT: [[casting_call_ref:%.*]] = lit.ref.struct.ger %[[SELF]][call]
# CHECK-NEXT: [[casting_call:%.*]] = lit.ref.load [[casting_call_ref]]
# CHECK-NEXT: [[result_of_typed_call:%.*]] = lit.call_signature [[casting_call]][{{.*}}](%[[RES]], [[closure_impl]], %n)
# CHECK-NEXT: lit.return [[result_of_typed_call]] : !kgen.none


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()

    fn __len__(self) -> Int:
        return 0


# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: MemType):
    fn myclosure(n: MemType) escaping -> MemType:
        return n + m

    fn myclosure2(n: MemType, j: Int) escaping -> Int:
        return m.__len__()
