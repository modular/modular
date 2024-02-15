# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


# CHECK-LABEL: lit.globalvar.decl @inferred_type : index
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@inferred_type : <index
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <1>
# CHECK-NEXT: lit.ref.store %[[VAL]], %[[REF]]
var inferred_type = `1`


# COM: this also serves for testing how we emit memory-only globals.
@value
struct ConvertibleFromInt:
    fn __init__(inout self, v: int):
        pass

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.globalvar.decl @conv_from_int : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@conv_from_int : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <2>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[REF]], %[[VAL]])
# CHECK: }, {
# CHECK-NEXT: %[[REF:.*]] = lit.globalvar.ref {{.*}}@conv_from_int
# CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%[[REF]])
let conv_from_int: ConvertibleFromInt = `2`

# CHECK-LABEL: lit.globalvar.decl @conv_from_int_implicit : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@conv_from_int_implicit : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <3>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[REF]], %[[VAL]])
let conv_from_int_implicit = ConvertibleFromInt(`3`)


@value
@register_passable
struct RegType:
    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.globalvar.decl @reg_global : !RegType
# CHECK-DAG: %[[VAL:.*]] = lit.call {{.*}}@RegType::@"__init__()"()
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@reg_global
# CHECK-NEXT: lit.ref.store %[[VAL]], %[[REF]]
# CHECK: }, {
# CHECK-NEXT: %[[REF:.*]] = lit.globalvar.ref {{.*}}@reg_global
# CHECK-NEXT: %[[CONSUMED:.*]] = lit.load.consume %[[REF]]
# CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%[[CONSUMED]])
let reg_global: RegType = RegType()

# CHECK-LABEL: lit.globalvar.decl @reg_global_implicit : !RegType isVar
# CHECK-DAG: %[[VAL:.*]] = lit.call {{.*}}@RegType::@"__init__()"()
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@reg_global_implicit
# CHECK-NEXT: lit.ref.store %[[VAL]], %[[REF]]
var reg_global_implicit = RegType()


fn borrowGlobalInt(x: int):
    pass


fn borrowGlobalReg(x: RegType):
    pass


fn mutGlobalReg(inout x: RegType):
    pass


fn copyGlobalMem(owned x: ConvertibleFromInt):
    pass


fn refGlobals():
    # CHECK: %[[TRIVIAL:.*]] = lit.globalvar.ref {{.*}}@inferred_type
    # CHECK-NEXT: %[[VALUE:.*]] = lit.ref.load %[[TRIVIAL]]
    # CHECK-NEXT: call {{.*}}borrowGlobalInt{{.*}}(%[[VALUE]])
    borrowGlobalInt(inferred_type)

    # CHECK: [[REG:%.*]] = lit.globalvar.ref {{.*}}@reg_global
    # CHECK-NEXT: %[[VALUE:.*]] = lit.ref.load [[REG]]
    # CHECK-NEXT: call {{.*}}borrowGlobalReg{{.*}}(%[[VALUE]])
    borrowGlobalReg(reg_global)

    # CHECK: %[[REG_REF:.*]] = lit.globalvar.ref {{.*}}@reg_global
    # CHECK-NEXT: call {{.*}}mutGlobalReg{{.*}}(%[[REG_REF]])
    mutGlobalReg(reg_global_implicit)

    # CHECK: %[[MEM_REF:.*]] = lit.globalvar.ref {{.*}}@conv_from_int
    # CHECK-NEXT: %anonymous2A = lit.varlet.decl {{.*}} : !lit.ref<!ConvertibleFromInt
    # CHECK-NEXT: %[[MEM_REF_IMM:.*]] = lit.ref.immut %[[MEM_REF]]
    # CHECK-NEXT: call {{.*}}__copyinit__{{.*}}(%anonymous2A, %[[MEM_REF_IMM]])
    # CHECK-NEXT: call {{.*}}copyGlobalMem{{.*}}(%anonymous2A)
    copyGlobalMem(conv_from_int)
