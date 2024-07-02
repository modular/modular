# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |, %q: !Int owned, %ww: !Int)
# CHECK-NEXT: %[[V0:.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: kgen.param.declare *"x`2": lifetime<1> = <(mutcast imm *"self`")>
# CHECK-NEXT: %[[V0REF:.*]] = kgen.rebind %[[V0]]
# CHECK-NEXT: %[[V1:.*]] = lit.ref.struct.ger %[[SELF]][field1]
# CHECK-NEXT: kgen.param.declare *"w`1": lifetime<1> = <(mutcast imm *"self`")>
# CHECK-NEXT: %[[V1REF:.*]] = kgen.rebind %[[V1]]
# CHECK-NEXT: %q_0 = lit.var.decl "q" arg
# CHECK-NEXT: lit.ref.store %q, %q_0
# CHECK-NEXT: %[[V2:.*]] = lit.ref.load %[[V0REF]]
# CHECK-NEXT: %[[V3:.*]] = lit.ref.load %[[V0REF]]
# CHECK-NEXT: %[[V4:.*]] = lit.call @{{.*}}::@Int::@"__add__{{.*}}"(%[[V2]], %[[V3]])
# CHECK-NEXT: lit.ref.store %[[V4]], %[[V0REF]]
# CHECK-NEXT: %[[V5:.*]] = lit.ref.load %[[V1REF]]
# CHECK-NEXT: lit.call @{{.*}}@"use{{.*}}"(%[[V5]])
# CHECK-NEXT: kgen.param.constant: none
# CHECK-NEXT: lit.return

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |, %p: !Int) -> !kgen.none
# CHECK-NEXT: %[[W0:.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: kgen.param.declare *"m`": lifetime<0> = <*"self`">
# CHECK-NEXT: %[[W0CAST:.*]] = kgen.rebind %[[W0]] : !lit.ref<!MemType, imm *"self`"> to !lit.ref<!MemType, imm *"m`">
# CHECK-NEXT: lit.call @{{.*}}::@"use{{.*}}(%[[W0CAST]])
# CHECK-NEXT: kgen.param.constant: none
# CHECK-NEXT: lit.return

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |) -> index
# CHECK-NEXT: %[[W0:.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: %[[W1:.*]] = lit.ref.load %[[W0]]
# CHECK-NEXT: %[[W2:.*]] = lit.ref.struct.ger %[[SELF]][field1]
# CHECK-NEXT: kgen.param.declare *"w`1": lifetime<1> = <(mutcast imm *"self`")>
# CHECK-NEXT: %[[W2REF:.*]] = kgen.rebind %[[W2]]
# CHECK-NEXT: %[[W3:.*]] = lit.ref.struct.ger %[[W2REF]][value]
# CHECK-NEXT: %[[W4:.*]] = lit.ref.load %[[W3]]
# CHECK-NEXT: %[[W5:.*]] = index.mul %[[W1]], %[[W4]]
# CHECK-NEXT: lit.return %[[W5]] : index

# CHECK: lit.func @"__call__{{.*}}_CI_{{.*}}(%[[SELF:.*]][{{.*}}]: !lit.ref<{{.*}}> borrow_in_mem, |, %y: !lit.ref<{{.*}}> borrow_in_mem, ?, [[RESULT:%.*]]: !lit.ref<!MemType,{{.*}}> byref_result) -> !kgen.none
# CHECK-NEXT: %[[W0:.*]] = lit.ref.struct.ger %[[SELF]][field0]
# CHECK-NEXT: kgen.param.declare *"m`"
# CHECK-NEXT: %[[W0REF:.*]] = kgen.rebind %[[W0]]
# CHECK-NEXT: lit.call @{{.*}}__add__{{.*}}(%[[W0REF]], %y, [[RESULT]])
# CHECK-NEXT: kgen.param.constant: none
# CHECK-NEXT: lit.return


fn use(x: MemType):
    pass


fn use(x: Int):
    pass


fn make_diff_closures(m: MemType, z: __mlir_type.index, owned w: Int):
    var x = w

    fn ret_mem(y: MemType) -> MemType:
        return m + y

    fn ret_mlir_type() -> __mlir_type.index:
        return __mlir_op.`index.mul`(z, w.value)

    fn ret_none(p: Int):
        use(m)

    fn capture_slvalue(owned q: Int, ww: Int):
        x = x + x
        use(w)
