# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.struct.decl @"fn
# CHECK-NEXT: destructor :!lit.generator
# CHECK-NEXT: move :!lit.generator
# CHECK-NEXT: copy :!lit.generator

# CHECK-LABEL: lit.fn @"__init__{{.*}}(%impl: {{.*}} owned_in_mem, |, ?, %self: !lit.ref<!MemType1, mut *"self`"> byref_result)
# CHECK-NEXT: %[[ptrToCall:.*]] = kgen.create_closure[!lit.generator<[2](!kgen.pointer<none>, |, "n": !lit.ref<!MemType, {{.*}}> read_mem, ?, "__result__": !lit.ref<!MemType, {{.*}}> byref_result) -> !kgen.none
# CHECK-NEXT: %[[callPtr:.*]] = lit.ref.struct.ger %self[call]
# CHECK-NEXT: lit.ref.store %[[ptrToCall]], %[[callPtr]]

# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: %[[V5:.*]] = lit.ref.struct.ger %self[dtor]
# CHECK-NEXT: lit.ref.store %[[V6]], %[[V5]]

# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: %[[V9:.*]] = lit.ref.struct.ger %self[_copy]
# CHECK-NEXT: lit.ref.store %[[V10]], %[[V9]]

# Allocate memory on heap
# CHECK-NEXT:  %[[SIZEOF:.*]] = kgen.param.constant = <get_sizeof({{.*}}escaping0{{.*}}, current_target())>
# CHECK-NEXT:  %[[ALIGNOF:.*]] = kgen.param.constant = <get_alignof({{.*}}escaping0{{.*}}, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %[[ALIGNOF]], %[[SIZEOF]]

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V0REF:.*]] = lit.ref.from_pointer %[[V0]]
# CHECK-NEXT:  %[[V1:.*]] = lit.call {{.*}}__moveinit__{{.*}}(%impl, %[[V0REF]])

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !kgen.pointer<{{.*}}> to !kgen.pointer<none>
# CHECK-NEXT:  %[[V2:.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:  lit.ref.store %[[V3]], %[[V2]]

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:  lit.return %[[V4]] : !kgen.none
# CHECK-NEXT:  lit.end_fn
# CHECK-NEXT:  }

# CHECK: lit.struct.decl @MemType

# CHECK-LABEL: lit.fn @"fn{{.*}}_dtor_`_CI_{{.*}}(%self: !kgen.pointer<none>, |)
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: %1 = lit.ref.from_pointer %0 end_uninit
# CHECK-NEXT: pop.aligned_free %0

# CHECK-LABEL: lit.fn @"fn{{.*}}_call_`_CI_{{.*}}](
# CHECK-SAME: %[[SELF:.*]][{{.*}}]: !kgen.pointer<none>, |, %n: !lit.ref<!MemType, imm {{.*}}> read_mem
# CHECK-SAME: %[[RES:.*]]: !lit.ref<!MemType, mut {{.*}}> byref_result)
# CHECK-NEXT: %[[A0:.*]] = pop.pointer.bitcast %[[SELF]]
# CHECK-NEXT: %[[A0REF:.*]] = lit.ref.from_pointer %[[A0]]
# CHECK-NEXT: lit.call {{.*}}__call__{{.*}}(%[[A0REF]], %n, %[[RES]])
# CHECK-NEXT: lit.return


@fieldwise_init
struct MemType(Copyable, Movable):
    pass


fn use(x: MemType):
    pass


fn thing(m: MemType):
    fn nested(n: MemType) -> MemType:
        use(m)
        return n
