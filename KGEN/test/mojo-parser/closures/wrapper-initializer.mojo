# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s


# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: copyInit =
# CHECK-SAME: destructor =
# CHECK-SAME: moveInit =

# CHECK-LABEL: lit.func @"__init__{{.*}}(%self: !lit.ref<mut !wrapper, *"self`"> init_self,
# CHECK-SAME: %impl: !lit.ref<mut !escaping0_, {{.*}}> owned_in_mem, |)
# CHECK-NEXT: %[[callPtr:.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT: %[[ptrToCall:.*]] = kgen.create_closure[!lit.signature<[2](!lit.ref<mut !MemType, {{.*}}> byref_result, !kgen.pointer<none> borrow, |, "n": !lit.ref<!MemType, {{.*}}> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: lit.ref.store %[[ptrToCall]], %[[callPtr]]

# CHECK-NEXT: %[[V5:.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: lit.ref.store %[[V6]], %[[V5]]

# CHECK-NEXT: %[[V9:.*]] = lit.ref.struct.ger %self[copy]
# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: lit.ref.store %[[V10]], %[[V9]]

# Allocate memory on heap
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof({{.*}}escaping0{{.*}}, current_target())>
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof({{.*}}escaping0{{.*}}, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %index_0, %index

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V0REF:.*]] = lit.ref.from_pointer %[[V0]]
# CHECK-NEXT:  %[[V1:.*]] = lit.call {{.*}}__moveinit__{{.*}}(%[[V0REF]], %impl)

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V2:.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !kgen.pointer<{{.*}}> to !kgen.pointer<none>
# CHECK-NEXT:  lit.ref.store %[[V3]], %[[V2]]

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:  lit.return %[[V4]] : !kgen.none
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }

# CHECK: lit.struct.decl @MemType

# CHECK-LABEL: lit.func @"_CW_{{.*}}_dtor_`_CI_{{.*}}(%self: !kgen.pointer<none>, |)
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: %1 = lit.ref.from_pointer %0 end_uninit
# CHECK-NEXT: pop.aligned_free %0

# CHECK-LABEL: lit.func @"_CW_{{.*}}_call_`_CI_{{.*}}[*"0_unnamed`", *"2_unnamed`"]
# CHECK-SAME: (%[[RES:.*]][{{.*}}]: !lit.ref<mut !MemType, {{.*}}> byref_result,
# CHECK-SAME: %[[SELF:.*]][{{.*}}]: !kgen.pointer<none> borrow, |, %n: !lit.ref<!MemType, {{.*}}> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: %[[A0:.*]] = pop.pointer.bitcast %[[SELF]]
# CHECK-NEXT: %[[A0REF:.*]] = lit.ref.from_pointer %[[A0]]
# CHECK-NEXT: lit.call {{.*}}__call__{{.*}}(%[[RES]], %[[A0REF]], %n)
# CHECK-NEXT: lit.return


@value
struct MemType:
    pass


fn use(x: MemType):
    pass


fn thing(m: MemType):
    fn nested(n: MemType) escaping -> MemType:
        use(m)
        return n
