# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%[[SELF:.*]][{{.*}}]: !kgen.pointer<{{.*}}> borrow_in_mem, |, %q[q]: !Int, %ww[ww]: !Int borrow) -> !kgen.none
# CHECK-NEXT: %[[V0:.*]] = lit.struct.gep %[[SELF]][field0] : <!Int>
# CHECK-NEXT: %[[V0REF:.*]] = builtin.unrealized_conversion_cast %[[V0]]
# CHECK-NEXT: %[[V1:.*]] = lit.struct.gep %[[SELF]][field1] : <!Int>
# CHECK-NEXT: %[[V1REF:.*]] = builtin.unrealized_conversion_cast %[[V1]]
# CHECK-NEXT: %q_0 = lit.varlet.decl "q" imp
# CHECK-NEXT: lit.ref.store %q, %q_0
# CHECK-NEXT: %[[V2:.*]] = lit.ref.load %[[V0REF]]
# CHECK-NEXT: %[[V3:.*]] = lit.ref.load %[[V0REF]]
# CHECK-NEXT: %[[V4:.*]] = lit.call @{{.*}}::@Int::@"__add__{{.*}}"(%[[V2]], %[[V3]]) : !lit.signature<("self": !Int borrow, "rhs": !Int borrow) -> !Int>
# CHECK-NEXT: lit.ref.store %[[V4]], %[[V0REF]]
# CHECK-NEXT: %[[V5:.*]] = lit.ref.load %[[V1REF]]
# CHECK-NEXT: %[[V6:.*]] = lit.call @{{.*}}@"use{{.*}}"(%[[V5]])
# CHECK-NEXT: %[[V7:.*]] = kgen.param.constant: none
# CHECK-NEXT: lit.return %[[V7]] : !kgen.none
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%[[SELF:.*]][{{.*}}]: !kgen.pointer<{{.*}}> borrow_in_mem, |, %p[p]: !Int borrow) -> !kgen.none
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %[[SELF]][field0] : <!MemType>
# CHECK-NEXT: %[[W1:.*]] = lit.call @{{.*}}::@"use{{.*}}"(%[[W0]])
# CHECK-NEXT: %[[W2:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %[[W2]] : !kgen.none
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%[[SELF:.*]][{{.*}}]: !kgen.pointer<{{.*}}> borrow_in_mem, |) -> index
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %[[SELF]][field0]
# CHECK-NEXT: %[[W1:.*]] = pop.load %[[W0]]
# CHECK-NEXT: %[[W2:.*]] = lit.struct.gep %[[SELF]][field1]
# CHECK-NEXT: %[[W2REF:.*]] = builtin.unrealized_conversion_cast %[[W2]]
# CHECK-NEXT: %[[W3:.*]] = lit.ref.struct.ger %[[W2REF]][value]
# CHECK-NEXT: %[[W4:.*]] = lit.ref.load %[[W3]]
# CHECK-NEXT: %[[W5:.*]] = index.mul %[[W1]], %[[W4]]
# CHECK-NEXT: lit.return %[[W5]] : index
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__{{.*}}_CI_{{.*}}"[{{.*}}](%0[{{.*}}]: !lit.ref<mut !MemType,{{.*}}> byref_result, %[[SELF:.*]][{{.*}}]: !kgen.pointer<{{.*}}> borrow_in_mem, |, %y[y]: !kgen.pointer<!MemType> borrow_in_mem) -> !kgen.none
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %[[SELF]][field0] : <!MemType>
# CHECK-NEXT: %[[W2:.*]] = lit.call @{{.*}}__add__{{.*}}(%0, %[[W0]], %y)
# CHECK-NEXT: %[[W4:.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %[[W4]] : !kgen.none
# CHECK-NEXT: lit.end_func


fn use(x: MemType):
    pass


fn use(x: Int):
    pass


fn make_diff_closures(m: MemType, z: __mlir_type.index, owned w: Int):
    var x = w

    fn ret_mem(y: MemType) escaping -> MemType:
        return m + y

    fn ret_mlir_type() escaping -> __mlir_type.index:
        return __mlir_op.`index.mul`(z, w.value)

    fn ret_none(p: Int) escaping:
        use(m)

    fn capture_slvalue(owned q: Int, ww: Int) escaping:
        x = x + x
        use(w)
