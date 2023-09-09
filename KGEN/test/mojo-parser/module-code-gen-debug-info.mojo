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
# CHECK-DAG: %[[VAL0:.*]] = lit.struct.gep %self[dtor] {{.*}} loc(#[[LOC_DEL:loc[0-9]*]])
# CHECK-DAG: %[[VAL1:.*]] = pop.load %[[VAL0]] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: %[[VAL2:.*]] = lit.struct.gep %self[field0] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: %[[VAL3:.*]] = pop.load %[[VAL2]] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: } loc(#[[LOC_DEL]])

# CHECK-DAG: lit.func @"__copyinit__(${{.*}}::_CW_
# CHECK-DAG: %[[W0:.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> loc(#[[LOC_COPY:loc[0-9]*]])
# CHECK-DAG: %[[W1:.*]] = lit.struct.gep %existing[field0] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W2:.*]] = pop.load %[[W0]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W3:.*]] = pop.load %[[W1]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W4:.*]] = lit.struct.gep %self[copy] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W5:.*]] = pop.load %[[W4]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: kgen.call_signature %[[W5]](%[[W2]], %[[W3]]) {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: } loc(#[[LOC_COPY]])

# CHECK-DAG: lit.func @"__moveinit__(${{.*}}::_CW_
# CHECK-DAG: %[[V0:.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> loc(#[[LOC_MOV:loc[0-9]*]])
# CHECK-DAG: %[[V1:.*]] = lit.struct.gep %existing[field0] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V2:.*]] = pop.load %[[V0]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V3:.*]] = pop.load %[[V1]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V4:.*]] = lit.struct.gep %self[move] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V5:.*]] = pop.load %[[V4]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: kgen.call_signature %[[V5]](%[[V2]], %[[V3]]) {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: } loc(#[[LOC_MOV]])

# CHECK-DAG: #[[LOC_DEL]] = loc(fused<#[[SP1]]>[#[[LOC:loc[0-9]*]]])
# CHECK-DAG: #[[LOC_MOV]] = loc(fused<#[[SP2]]>[#[[LOC]]])
# CHECK-DAG: #[[LOC_COPY]] = loc(fused<#[[SP3]]>[#[[LOC]]])

fn makes_escaping_closure(m:  __mlir_type.index, z: __mlir_type.index) -> fn(n:__mlir_type.index) escaping ->  __mlir_type.index:
   fn myclosure(n: __mlir_type.index) escaping ->  __mlir_type.index:
      return m
   return myclosure

# // -----

# CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__del__", linkageName = "__del__{{.*}}::_CI_{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR4:.*]]
# CHECK-DAG: #[[SP2:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__moveinit__", linkageName = "__moveinit__(${{.*}}::_CI_${{.*}}(${{.*}}::InMemType,__mlir_type.index){{.*}}=&,${{.*}}::_CI_${{.*}}(${{.*}}::InMemType,__mlir_type.index){{.*}})", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR5:.*]]
# CHECK-DAG: #[[SP3:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__copyinit__", linkageName = "__copyinit__{{.*}}::_CI_{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : ![[SR5]]

# CHECK-DAG: lit.ownership.mark.destroyed %self : !kgen.pointer<@{{.*}}::@"_CI_{{.*}}(${{.*}}::InMemType,__mlir_type.index)\22"> loc(#[[CI_LOC_DEL:.*]])
# CHECK-DAG: lit.ownership.mark.destroyed %existing : !kgen.pointer<@{{.*}}::@"_CI_{{.*}}(${{.*}}::InMemType,__mlir_type.index)\22"> loc(#[[CI_LOC_MOV:.*]])

# CHECK-DAG: #[[CI_LOC_DEL]] = loc(fused<#[[SP1]]>[#[[CI_LOC:.*]]])
# CHECK-DAG: #[[CI_LOC_MOV]] = loc(fused<#[[SP2]]>[#[[CI_LOC]]])

@value
struct InMemType:
   fn __del__(owned self):
       pass

fn makes_escaping_closure(m:  InMemType, z: __mlir_type.index):
   fn dummy(n: __mlir_type.index) escaping ->  InMemType:
      return m

# // -----

fn makes_escaping_closure(z: Int):
   let w = z * z
   var a = w
   # CHECK-DAG: %anonymous2A = lit.varlet.decl "anonymous*" var synth : <@"{{.*}}_CI_{{.*}}({{.*}}::Int,{{.*}}::Int,{{.*}}::Int)\22"> loc(#[[LOC:loc[0-9]*]])
   # CHECK-DAG: %[[A:.*]] = pop.load %a : !kgen.pointer<!Int> loc(#[[LOC]])
   # CHECK-DAG: kgen.call @{{.*}}::@"__init__{{.*}}"(%anonymous2A, %[[A]], %w) : ("self": !kgen.pointer<@"{{.*}}"::@"_CI_${{.*}}($builtin::$int::Int,$builtin::$int::Int,$builtin::$int::Int){{.*}}"> init_self, "field0": !Int, "field1": !Int) -> !lit.none loc(#[[LOC]])
   fn myclosure_with_reg_types(x:Int) escaping -> Int:
      a = a + 1
      return x + w

# CHECK-DAG: #[[LOC]] = loc(fused<#subprogram[[ID:.*]]>[#loc[[LOC2:.*]]])
# CHECK-DAG: #subprogram[[ID]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "makes_escaping_closure", linkageName = "makes_escaping_closure($builtin::$int::Int)", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : !subroutine[[ID]]
