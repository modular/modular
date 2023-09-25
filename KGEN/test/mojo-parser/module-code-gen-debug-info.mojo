# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -verify-diagnostics -import-mojo -debug-level full -mlir-print-debuginfo -split-input-file %s | FileCheck %s

# CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__del__", linkageName = "__del__{{.*}}_CW_{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR1:.*]]
# CHECK-DAG: #[[SP2:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__moveinit__", linkageName = "__moveinit__{{.*}}_CW_{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR2:.*]]
# CHECK-DAG: #[[SP3:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__copyinit__", linkageName = "__copyinit__{{.*}}_CW_{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR2]]


# CHECK-DAG: lit.func @"__del__(${{.*}}::_CW_
# CHECK-DAG: %[[VAL0:.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_DEL:loc[0-9]*]])
# CHECK-DAG: %[[VAL1:.*]] = pop.load %[[VAL0]] : !kgen.pointer<pointer<array<0, i1>>> loc(#[[LOC_DEL]])
# CHECK-DAG: } loc(#[[LOC_DEL]])

# CHECK-DAG: lit.func @"__copyinit__(${{.*}}::_CW_
# CHECK-DAG:   [[M0:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_COPY:loc[0-9]*]])
# CHECK-DAG:   [[other_impl:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_COPY]])
# CHECK-DAG:   [[loaded_other_impl:%.*]] = pop.load [[other_impl]] : !kgen.pointer<pointer<array<0, i1>>> loc(#[[LOC_COPY]])
# CHECK-DAG:   pop.store [[loaded_other_impl]], [[M0]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M1:%.*]] = lit.struct.gep %self[dtor] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M2:%.*]] = lit.struct.gep %existing[dtor] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M3:%.*]] = pop.load [[M2]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   pop.store [[M3]], [[M1]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M4:%.*]] = lit.struct.gep %self[copy] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M5:%.*]] = lit.struct.gep %existing[copy] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M6:%.*]] = pop.load [[M5]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   pop.store [[M6]], [[M4]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M7:%.*]] = lit.struct.gep %self[call] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M8:%.*]] = lit.struct.gep %existing[call] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   [[M9:%.*]] = pop.load [[M8]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   pop.store [[M9]], [[M7]] {{.*}} loc(#[[LOC_COPY]])

# CHECK-DAG:   %[[W0:.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_COPY]])
# CHECK-DAG:   %[[W1:.*]] = pop.load %[[W0]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   %[[W2:.*]] = lit.struct.gep %self[copy] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   %[[W3:.*]] = lit.struct.gep %self[field0] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   %[[W4:.*]] = pop.load %[[W2]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG:   kgen.call_signature %[[W4]](%[[W3]], %[[W1]]) {{.*}} loc(#[[LOC_COPY]])

# CHECK-DAG: lit.func @"__moveinit__(${{.*}}::_CW_
# CHECK-DAG:   [[R0:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_MOV:loc[0-9]*]])
# CHECK-DAG:   [[mov_other_impl:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_MOV]])
# CHECK-DAG:   [[loaded_mov_other_impl:%.*]] = lit.load.consume [[mov_other_impl]] : !kgen.pointer<pointer<array<0, i1>>> loc(#[[LOC_MOV]])
# CHECK-DAG:   pop.store [[loaded_mov_other_impl]], [[R0]] : !kgen.pointer<pointer<array<0, i1>>> loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R1:%.*]] = lit.struct.gep %self[dtor] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R2:%.*]] = lit.struct.gep %existing[dtor] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R3:%.*]] = lit.load.consume [[R2]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   pop.store [[R3]], [[R1]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R4:%.*]] = lit.struct.gep %self[copy] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R5:%.*]] = lit.struct.gep %existing[copy] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R6:%.*]] = lit.load.consume [[R5]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   pop.store [[R6]], [[R4]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R7:%.*]] = lit.struct.gep %self[call] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R8:%.*]] = lit.struct.gep %existing[call] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   [[R9:%.*]] = lit.load.consume [[R8]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   pop.store [[R9]], [[R7]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG:   %pointer = kgen.param.constant: pointer<array<0, i1>> = <0> loc(#[[LOC_MOV]])
# CHECK-DAG:   %[[V0:.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>> from <!escaping1> loc(#[[LOC_MOV]])
# CHECK-DAG:   pop.store %pointer, %[[V0]] : !kgen.pointer<pointer<array<0, i1>>> loc(#[[LOC_MOV]])

# CHECK-DAG: #[[LOC_DEL]] = loc(fused<#[[SP1]]>
# CHECK-DAG: #[[LOC_MOV]] = loc(fused<#[[SP2]]>
# CHECK-DAG: #[[LOC_COPY]] = loc(fused<#[[SP3]]>


fn capture_index(
    m: __mlir_type.index, z: __mlir_type.index
) -> fn (n: __mlir_type.index) escaping -> __mlir_type.index:
    fn myclosure(n: __mlir_type.index) escaping -> __mlir_type.index:
        return m

    return myclosure


# // -----

# CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__del__", linkageName = "__del__{{.*}} : ![[SR4:.*]]
# CHECK-DAG: #[[SP2:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__moveinit__", linkageName = "__moveinit__{{.*}} : ![[SR5:.*]]
# CHECK-DAG: #[[SP3:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__copyinit__", linkageName = "__copyinit__{{.*}} : ![[SR5]]

# CHECK-DAG: ![[ESCAPING:.*]] = !kgen.declref<{{.*}}CI{{.*}}InMemType{{.*}}>
# CHECK-DAG: lit.ownership.mark_destroyed %self : !kgen.pointer<![[ESCAPING]]> loc(#[[CI_LOC_DEL:.*]])
# CHECK-DAG: lit.ownership.mark_destroyed %existing : !kgen.pointer<![[ESCAPING]]> loc(#[[CI_LOC_MOV:.*]])

# CHECK-DAG: #[[CI_LOC_DEL]] = loc(fused<#[[SP1]]>[#[[CI_LOC:.*]]])
# CHECK-DAG: #[[CI_LOC_MOV]] = loc(fused<#[[SP2]]>[#[[CI_LOC]]])


@value
struct InMemType:
    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.func @"capture_memory_type
fn capture_memory_type(m: InMemType, z: __mlir_type.index):
    fn dummy(n: __mlir_type.index) escaping -> InMemType:
        return m


# // -----

# CHECK: #subprogram[[ID:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "capture_reg_type", linkageName = "capture_reg_type($builtin::$int::Int)", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : !subroutine


# CHECK: lit.func @"capture_reg_type
fn capture_reg_type(z: Int):
    let w = z * z
    var a = w

    # CHECK: %anonymous2A = lit.varlet.decl "anonymous*" var synth : !lit.ref<mut !escaping{{.*}} loc(#[[LOC:loc[0-9]*]])
    # CHECK-NEXT: [[ANONPTR:%.*]] = lit.ref.to_pointer %anonymous2A
    # CHECK: [[A:%.*]] = lit.ref.load %a : <mut !Int, {{.*}} loc(#[[LOC]])
    # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"([[ANONPTR]], [[A]], %w) {{.*}} loc(#[[LOC]])
    fn myclosure_with_reg_types(x: Int) escaping -> Int:
        a = a + 1
        return x + w


# CHECK: #[[LOC]] = loc(fused<#subprogram[[ID]]>[#loc[[LOC2:.*]]])
