# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -verify-diagnostics -import-mojo -debug-level=full -mlir-print-debuginfo %s | FileCheck %s

# CHECK-DAG: #subprogram = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__del__{{.*}}", linkageName = "__del__{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : !subroutine
# CHECK-DAG: #subprogram1 = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__moveinit__{{.*}}", linkageName = "__moveinit__{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : !subroutine1
# CHECK-DAG: #subprogram2 = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "__copyinit__{{.*}}", linkageName = "__copyinit__{{.*}}", file = #file, line = {{.*}}, scopeLine = {{.*}}, subprogramFlags = "Definition|Optimized"> : !subroutine1


# CHECK-DAG: lit.func @"__del__($module-code-gen-debug-info::_CW_
# CHECK-DAG: %[[VAL0:.*]] = lit.struct.gep %self[dtor] {{.*}} loc(#[[LOC_DEL:loc[0-9]+]])
# CHECK-DAG: %[[VAL1:.*]] = pop.load %[[VAL0]] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: %[[VAL2:.*]] = lit.struct.gep %self[field0] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: %[[VAL3:.*]] = pop.load %[[VAL2]] {{.*}} loc(#[[LOC_DEL]])
# CHECK-DAG: } loc(#[[LOC_DEL]])

# CHECK-DAG: lit.func @"__moveinit__($module-code-gen-debug-info::_CW_
# CHECK-DAG: %[[V0:.*]] = lit.struct.gep %self[field0] {{.*}} loc(#[[LOC_MOV:loc[0-9]+]])
# CHECK-DAG: %[[V1:.*]] = lit.struct.gep %existing[field0] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V2:.*]] = pop.load %[[V0]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V3:.*]] = pop.load %[[V1]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V4:.*]] = lit.struct.gep %self[move] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: %[[V5:.*]] = pop.load %[[V4]] {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: kgen.call_signature %[[V5]](%[[V2]], %[[V3]]) {{.*}} loc(#[[LOC_MOV]])
# CHECK-DAG: } loc(#[[LOC_MOV]])

# CHECK-DAG: lit.func @"__copyinit__($module-code-gen-debug-info::_CW_
# CHECK-DAG: %[[W0:.*]] = lit.struct.gep %self[field0] {{.*}} loc(#[[LOC_COPY:loc[0-9]+]])
# CHECK-DAG: %[[W1:.*]] = lit.struct.gep %existing[field0] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W2:.*]] = pop.load %[[W0]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W3:.*]] = pop.load %[[W1]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W4:.*]] = lit.struct.gep %self[copy] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: %[[W5:.*]] = pop.load %[[W4]] {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: kgen.call_signature %[[W5]](%[[W2]], %[[W3]]) {{.*}} loc(#[[LOC_COPY]])
# CHECK-DAG: } loc(#[[LOC_COPY]])

# CHECK-DAG: #[[LOC_DEL]] = loc(fused<#subprogram>[#loc1])
# CHECK-DAG: #[[LOC_MOV]] = loc(fused<#subprogram1>[#loc1])
# CHECK-DAG: #[[LOC_COPY]] = loc(fused<#subprogram2>[#loc1])

fn makes_escaping_closure(m:  __mlir_type.index, z: __mlir_type.index) -> fn( __mlir_type.index) escaping ->  __mlir_type.index:
   fn myclosure(n: __mlir_type.index) ->  __mlir_type.index:
      return m
   return myclosure
