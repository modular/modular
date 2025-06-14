# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.globalvar.decl @__inferred_type : index
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@__inferred_type : <index
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <1>
# CHECK-NEXT: lit.ref.store %[[VAL]], %[[REF]]
var __inferred_type = `1`


# COM: this also serves for testing how we emit memory-only globals.
@fieldwise_init
struct ConvertibleFromInt(Copyable, Movable):
    @implicit
    fn __init__(out self, v: Index):
        pass

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.globalvar.decl @__conv_from_int : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@__conv_from_int : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <2>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[VAL]], %[[REF]])
# CHECK: }, {
# CHECK-NEXT: %[[REF:.*]] = lit.globalvar.ref {{.*}}@__conv_from_int
# CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%[[REF]])
var __conv_from_int: ConvertibleFromInt = `2`

# CHECK-LABEL: lit.globalvar.decl @__conv_from_int_implicit : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@__conv_from_int_implicit : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <3>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[VAL]], %[[REF]])
var __conv_from_int_implicit = ConvertibleFromInt(`3`)


@fieldwise_init
@register_passable
struct RegType(Copyable):
    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.globalvar.decl @__reg_global : !RegType
# CHECK-NEXT: [[VAL:%.*]] = lit.call {{.*}}RegType::@"__init__{{.*}}()
# CHECK-NEXT: [[REF:%.*]] = lit.globalvar.ref {{.*}}@__reg_global
# CHECK-NEXT: lit.ref.store [[VAL]], [[REF]]

# CHECK: }, {
# CHECK-NEXT: %[[REF:.*]] = lit.globalvar.ref {{.*}}@__reg_global
# CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%[[REF]])
var __reg_global: RegType = RegType()

# CHECK-LABEL: lit.globalvar.decl @__reg_global_implicit : !RegType
# CHECK-NEXT: [[VAL:%.*]] = lit.call {{.*}}RegType::@"__init__{{.*}}()
# CHECK-NEXT: [[REF:%.*]] = lit.globalvar.ref {{.*}}@__reg_global_implicit
# CHECK-NEXT: lit.ref.store [[VAL]], [[REF]]
var __reg_global_implicit = RegType()


fn borrowGlobalInt(x: Index):
    pass


fn borrowGlobalReg(x: RegType):
    pass


fn mutGlobalReg(mut x: RegType):
    pass


fn copyGlobalMem(owned x: ConvertibleFromInt):
    pass


fn refGlobals():
    # CHECK: %[[TRIVIAL:.*]] = lit.globalvar.ref {{.*}}@__inferred_type
    # CHECK-NEXT: %[[VALUE:.*]] = lit.ref.load %[[TRIVIAL]]
    # CHECK-NEXT: call {{.*}}borrowGlobalInt{{.*}}(%[[VALUE]])
    borrowGlobalInt(__inferred_type)

    # CHECK: [[REG:%.*]] = lit.globalvar.ref {{.*}}@__reg_global
    # CHECK-NEXT: %[[VALUE:.*]] = lit.ref.immut [[REG]]
    # CHECK-NEXT: call {{.*}}borrowGlobalReg{{.*}}(%[[VALUE]])
    borrowGlobalReg(__reg_global)

    # CHECK: %[[REG_REF:.*]] = lit.globalvar.ref {{.*}}@__reg_global
    # CHECK-NEXT: call {{.*}}mutGlobalReg{{.*}}(%[[REG_REF]])
    mutGlobalReg(__reg_global_implicit)

    # CHECK: %[[MEM_REF:.*]] = lit.globalvar.ref {{.*}}@__conv_from_int
    # CHECK-NEXT: %anonymous2A = lit.var.decl {{.*}} : !lit.ref<!ConvertibleFromInt
    # CHECK-NEXT: %[[MEM_REF_IMM:.*]] = lit.ref.immut %[[MEM_REF]]
    # CHECK-NEXT: call {{.*}}__copyinit__{{.*}}(%[[MEM_REF_IMM]], %anonymous2A)
    # CHECK-NEXT: call {{.*}}copyGlobalMem{{.*}}(%anonymous2A)
    copyGlobalMem(__conv_from_int)
