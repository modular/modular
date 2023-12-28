# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: copyInit =
# CHECK-SAME: destructor =
# CHECK-SAME: moveInit =

# CHECK: lit.func @"__init__{{.*}}"(%self[self]: !kgen.pointer<!wrapper> init_self, %impl[impl]: !kgen.pointer<!escaping0_> owned_in_mem, |)
# CHECK-NEXT: %[[callPtr:.*]] = lit.struct.gep %self[call]
# CHECK-NEXT: %[[ptrToCall:.*]] = kgen.create_closure[!lit.signature<[1](!lit.ref<mut !MemType, {{.*}}> byref_result, !kgen.pointer<none> borrow, |, "n": !kgen.pointer<!MemType> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: pop.store %[[ptrToCall]], %[[callPtr]]

# CHECK-NEXT: %[[V5:.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: pop.store %[[V6]], %[[V5]] : !kgen.pointer<!lit.signature<("self": !kgen.pointer<none>, |) -> !kgen.none>

# CHECK-NEXT: %[[V9:.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure[{{.*}}]()
# CHECK-NEXT: pop.store %[[V10]], %[[V9]]

# Allocate memory on heap
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof(!escaping0_, current_target())>
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof(!escaping0_, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %index_0, %index : <!escaping0_>

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V1:.*]] = lit.call {{.*}}__moveinit__{{.*}}(%[[V0]], %impl)

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V2:.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !kgen.pointer<!escaping0_> to !kgen.pointer<none>
# CHECK-NEXT:  pop.store %[[V3]], %[[V2]] : !kgen.pointer<pointer<none>>

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:  lit.return %[[V4]] : !kgen.none
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }

# CHECK: lit.struct.decl @MemType

# CHECK: lit.func @"_CW_{{.*}}_dtor_`_CI_{{.*}}"(%self[self]: !kgen.pointer<none>, |) -> !kgen.none
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: lit.ownership.end_lifetime %0
# CHECK-NEXT: pop.aligned_free %0

# CHECK: lit.func @"_CW_{{.*}}_call_`_CI_{{.*}}"[{{.*}}](%[[RES:.*]][{{.*}}]: !lit.ref<mut !MemType, {{.*}}> byref_result, %[[SELF:.*]][{{.*}}]: !kgen.pointer<none> borrow, |, %n[n]: !kgen.pointer<!MemType> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: %[[A0:.*]] = pop.pointer.bitcast %[[SELF]]
# CHECK-NEXT: %[[A1:.*]] = lit.call @{{.*}}@"__call__{{.*}}"[{{.*}}](%[[RES]], %[[A0]], %n)
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
