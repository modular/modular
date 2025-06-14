# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.field call : !kgen.generator<!lit.generator<[1](!kgen.pointer<none>, |, "n": !lit.ref<!MemType, imm {{.*}}> read_mem, "j": !Int) -> !Int>>
# CHECK: lit.fn @"__call__{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> read_mem, |, %n: !lit.ref<!MemType, imm {{.*}}> read_mem, %j: !Int) -> !Int
# CHECK-NEXT: [[closure_impl_ref0:%.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: [[closure_impl0:%.*]] = lit.ref.load [[closure_impl_ref0]]
# CHECK-NEXT: [[casting_call_ref0:%.*]] = lit.ref.struct.ger %[[SELF]][call]
# CHECK-NEXT: [[casting_call0:%.*]] = lit.ref.load [[casting_call_ref0]]
# CHECK-NEXT: [[result_of_typed_call0:%.*]] = lit.call_indirect [[casting_call0]]{{.*}}([[closure_impl0]], %n, %j)
# CHECK-NEXT: lit.return [[result_of_typed_call0]] : !Int
# CHECK-NEXT: lit.end_fn
# CHECK-NEXT: }

# CHECK: lit.struct.field call : !kgen.generator<!lit.generator<[2](!kgen.pointer<none>, |, "n": !lit.ref<!MemType, imm {{.*}}> read_mem, ?, "__result__": !lit.ref<!MemType, mut *[0,1]> byref_result) -> !kgen.none>>
# CHECK: lit.fn @"__call__{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> read_mem, |, %n: !lit.ref<!MemType, imm {{.*}}> read_mem, ?, %[[RES:.*]]: !lit.ref<!MemType, mut {{.*}}> byref_result)
# CHECK-NEXT: [[closure_impl_ref:%.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: [[closure_impl:%.*]] = lit.ref.load [[closure_impl_ref]]
# CHECK-NEXT: [[casting_call_ref:%.*]] = lit.ref.struct.ger %[[SELF]][call]
# CHECK-NEXT: [[casting_call:%.*]] = lit.ref.load [[casting_call_ref]]
# CHECK-NEXT: [[result_of_typed_call:%.*]] = lit.call_indirect [[casting_call]][{{.*}}]([[closure_impl]], %n, %[[RES]])
# CHECK-NEXT: lit.return [[result_of_typed_call]] : !kgen.none


@fieldwise_init
struct MemType(Copyable, Movable):
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()

    fn __len__(self) -> Int:
        return 0


# CHECK-LABEL: lit.fn @"makes_escaping_closure
fn makes_escaping_closure(m: MemType):
    fn myclosure(n: MemType) -> MemType:
        return n + m

    fn myclosure2(n: MemType, j: Int) -> Int:
        return m.__len__()
