# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# CHECK: lit.struct.decl @"_CW_

# CHECK: lit.func @"__init__{{.*}}"(%self[self]: !kgen.pointer<!escaping1> init_self, %impl[impl]: !kgen.pointer<!escaping> borrow_in_mem, |)
# CHECK-NEXT: %[[callPtr:.*]] = lit.struct.gep %self[call]
# CHECK-NEXT: %[[ptrToCall:.*]] = kgen.create_closure [!lit.signature<(!kgen.pointer<!MemType> byref_result, !kgen.pointer<none> borrow_in_mem, |, "n": !kgen.pointer<!MemType> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: pop.store %[[ptrToCall]], %[[callPtr]]

# CHECK-NEXT: %[[V5:.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V6]], %[[V5]] : !kgen.pointer<!lit.signature<("self": !kgen.pointer<none>, |) -> !kgen.none>

# CHECK-NEXT: %[[V9:.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V10]], %[[V9]]

# Allocate memory on heap
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof(!escaping, current_target())>
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof(!escaping, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %index_0, %index : <!escaping>

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V1:.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%[[V0]], %impl)

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V2:.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !kgen.pointer<!escaping> to !kgen.pointer<none>
# CHECK-NEXT:  pop.store %[[V3]], %[[V2]] : !kgen.pointer<pointer<none>>

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:  lit.return %[[V4]] : !kgen.none
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }

# CHECK: lit.struct.decl @MemType

# CHECK: lit.func @"_CW_{{.*}}_dtor__CI_{{.*}}"(%self[self]: !kgen.pointer<none>, |) -> !kgen.none
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: pop.aligned_free %0

# CHECK: lit.func @"_CW_{{.*}}_call__CI_{{.*}}"(%[[RES:.*]][{{.*}}]: !kgen.pointer<!MemType> byref_result, %[[SELF:.*]][{{.*}}]: !kgen.pointer<none> borrow_in_mem, |, %n[n]: !kgen.pointer<!MemType> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: %[[A0:.*]] = pop.pointer.bitcast %[[SELF]]
# CHECK-NEXT: %[[A1:.*]] = kgen.call @{{.*}}@"__call__{{.*}}"(%[[RES]], %[[A0]], %n)
# CHECK-NEXT: lit.return %[[A1]] : !kgen.none
# CHECK-NEXT: lit.end_func


@value
struct MemType:
    pass


fn use(x: MemType):
    pass


fn thing(m: MemType):
    fn nested(n: MemType) escaping -> MemType:
        use(m)
        return n
