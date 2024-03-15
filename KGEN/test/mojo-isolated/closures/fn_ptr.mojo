# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s -debug-level full


struct Mem:
    pass


@register_passable
struct Reg:
    pass


# CHECK-LABEL: lit.struct.decl @"fn(Mem
# CHECK-LABEL: lit.func @"__init__
# CHECK-SAME: (%self: {{.*}}, %other: {{.*}}[1](!lit.ref<!Mem{{[0-9]*}}, imm *[0,0]> borrow_in_mem, |) -> !Int>
# CHECK-NEXT: [[OPAQUE:%.*]] = pop.pointer.bitcast %other
# CHECK-NEXT: [[FIELD0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT: store [[OPAQUE]], [[FIELD0]]
# CHECK-NEXT: [[DTOR:%.*]] = kgen.create_closure[{{.*}}__closure_wrapper_noop_dtor
# CHECK-NEXT: [[COPY:%.*]] = kgen.create_closure[{{.*}}__closure_wrapper_noop_copy
# CHECK-NEXT: [[DTOR_FIELD:%.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT: store [[DTOR]], [[DTOR_FIELD]]
# CHECK-NEXT: [[COPY_FIELD:%.*]] = lit.ref.struct.ger %self[copy]
# CHECK-NEXT: store [[COPY]], [[COPY_FIELD]]
# CHECK-NEXT: lit.func call_impl[[[LT:.*]]]([[FN_PTR:%.*]][*""]: !kgen.pointer<none> borrow, [[ARG:%.*]][*""]: !lit.ref<!Mem, [[LT]]> borrow_in_mem, |) -> !Int
# CHECK-NEXT:   [[CALLEE:%.*]] = pop.pointer.bitcast [[FN_PTR]]
# CHECK-NEXT:   [[RES:%.*]] = lit.call_signature [[CALLEE]][[[LT]]]([[ARG]])
# CHECK-NEXT:   lit.return [[RES]]
# CHECK-NEXT:   lit.end_func
# CHECK-NEXT: }
# CHECK-NEXT: [[CALL:%.*]] = kgen.create_closure[{{.*}}call_impl
# CHECK-NEXT: [[CALL_FIELD:%.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT: store [[CALL]], [[CALL_FIELD]]

# CHECK-LABEL: lit.struct.decl @"fn(Reg
# CHECK-LABEL: lit.func @"__init__
# CHECK-SAME: (%self: {{.*}}, %other: {{.*}}[1]({{.*}}"__result__": !lit.ref<!Mem, mut *[0,0]> byref_result) throws|ownedresult -> !kgen.variant<!Error, none>
# CHECK:      lit.func call_impl[mut [[LT:.*]]]([[FN_PTR:%.*]][*""]: !kgen.pointer<none> borrow, [[ARG:%.*]][*""]: !Reg borrow, |, ?, [[SLOT:%.*]]: !lit.ref<!Mem, mut [[LT]]> byref_result) throws|ownedresult
# CHECK:        [[RES:%.*]] = lit.call_signature %{{.*}}[mut [[LT]]]([[ARG]], [[SLOT]])
# CHECK-NEXT:   lit.return [[RES]]


fn top(y: fn (Reg) escaping raises -> Mem):
    fn fn_ptr(x: Mem) -> Int:
        return 0

    var x: fn (Mem) escaping -> Int = fn_ptr
