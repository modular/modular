# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -I %S -split-input-file -debug-level full -mlir-print-debuginfo %s | FileCheck %s
from imported_module import imported_fn

# Check that we properly generate functions that get resolved within other functions.
# This is mostly checking that the scope of the nested function is not another function.

# CHECK-DAG: #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #file, name = "test", linkageName = "test($parser-debuginfo::CalledStruct[{{.*}}param])"


struct CalledStruct[param: __mlir_type.index]:
    fn test(self):
        imported_fn()


fn callerFn[rows: __mlir_type.index](arg0: CalledStruct[rows]):
    return arg0.test()


# Check single file debug info generation.

# CHECK-DAG: ![[INT_TYPE:.*]] = !debuginfo.unresolved<!Int>
# CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(!Int, !Int) -> (!Int): DW_CC_normal>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "power", linkageName = "power({{.*}}$int::Int,{{.*}}$int::Int)", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK-DAG: #[[LHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "lhs", file = #{{.*}}, line = [[LN]], arg = 1> : ![[INT_TYPE]]
# CHECK-DAG: #[[RHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "rhs", file = #{{.*}}, line = [[LN]], arg = 2> : ![[INT_TYPE]]

# CHECK-LABEL: lit.func @"power({{.*}}$int::Int,{{.*}}$int::Int)"(
fn power(lhs: Int, rhs: Int) -> Int:
    # CHECK: debuginfo.value #[[LHS_VAR]] = %lhs
    # CHECK: debuginfo.value #[[RHS_VAR]] = %rhs
    return lhs


# // -----

# CHECK-DAG: #[[LOCAL_VAR_I:.*]] = #debuginfo.local_variable<scope = #[[FOR_SP:.*]], name = "i", {{.*}}, line = [[LN:[0-9]+]], arg = 1

# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 loc({{.*}}) = %idx0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: debuginfo.value #[[LOCAL_VAR_I]] = %arg0 : index
        # CHECK-NEXT: %idx1 = index.constant 1
        # CHECK-NEXT: %1 = index.add %arg0, %idx1 loc(#[[FOR_ADD_LOC:.*]])
        # CHECK-NEXT: hlcf.continue %1 : index loc(#[[FOR_YIELD_LOC:.*]])
        __mlir_op.`hlcf.continue`(__mlir_op.`index.add`(i, (1).value))

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type : __mlir_type.index, _region : "loop_body".value
    ]((0).value)


# // -----

from imported_module import VeryUniqueStruct

# CHECK-DAG: #[[FILE:file[0-9]+]] = #debuginfo.file<"[[FILENAME:.*imported_module.mojo]]" in

# CHECK-DAG: #[[LOCAL_VAR:local_variable[0-9]*]] = #debuginfo.local_variable<scope = #[[SP:subprogram[0-9]+]], name = "very_unique_arg", file = #[[FILE]],

# CHECK-DAG: lit.struct.decl @VeryUniqueStruct
# CHECK-DAG: lit.struct.field very_unique_field : index loc(#[[LOC:loc[0-9]+]])
# CHECK-DAG: lit.func @"very_unique_func($builtin::$int::Int)"(%very_unique_arg: !Int loc("[[FILENAME]]"
# CHECK-DAG: debuginfo.value #[[LOCAL_VAR]] = %very_unique_arg : !Int loc(#[[VALUE_LOC:loc[0-9]+]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])
# CHECK-DAG: #[[VALUE_LOC]] = loc(fused<#[[SP]]>[#[[LINE_LOC:loc[0-9]+]]])
# CHECK-DAG: #[[LINE_LOC]] = loc("[[FILENAME]]"

fn caller():
    let y = VeryUniqueStruct.very_unique_func(0)

# // -----

# COM: These test the locations in the body of synthesized constructors. Because
# COM: of the indeterministic order in which attributes may be printed, we need
# COM: to use CHECK-DAG and order the statements to ensure unique matchings.

# CHECK-DAG: #[[SP_INIT:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE:file[0-9]*]], name = "__init__", linkageName = "__init__($parser-debuginfo::MyValueStruct=&,__mlir_type.index)",
# CHECK-DAG: #[[SP_COPY:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "__copyinit__", linkageName = "__copyinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)",
# CHECK-DAG: #[[SP_MOVE:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "__moveinit__", linkageName = "__moveinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)",
# CHECK-DAG: #[[FILE]] = #debuginfo.file<"within split at [[FILENAME:.*parser-debuginfo.mojo]]:{{[0-9]+}} offset " in "/">

# CHECK-DAG: lit.func @"__init__($parser-debuginfo::MyValueStruct=&,__mlir_type.index)"
# CHECK-DAG:   pop.store %value, %[[VAL:.*]] : !kgen.pointer<index> loc(#[[INIT_LOC:loc[0-9]*]])
# CHECK-DAG:   %[[VAL]] = lit.struct.gep %self[value] : <index> from <!MyValueStruct> loc(#[[INIT_LOC]])

# CHECK-DAG: lit.func @"__copyinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)"
# CHECK-DAG:   %[[VAL2:.*]] = pop.load %[[VAL1:.*]] : !kgen.pointer<index> loc(#[[COPY_LOC:loc[0-9]*]])
# CHECK-DAG:   pop.store %[[VAL2]], %[[VAL0:.*]] : !kgen.pointer<index> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL0]] = lit.struct.gep %self[value] : <index> from <!MyValueStruct> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL1]] = lit.struct.gep %existing[value] : <index> from <!MyValueStruct> loc(#[[COPY_LOC]])

# CHECK-DAG: lit.func @"__moveinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)"
# CHECK-DAG:   %[[VAL3:.*]] = lit.load.consume %[[VAL4:.*]] : !kgen.pointer<index> loc(#[[MOVE_LOC:loc[0-9]*]])
# CHECK-DAG:   pop.store %[[VAL3]], %[[VAL5:.*]] : !kgen.pointer<index> loc(#[[MOVE_LOC]])
# CHECK-DAG:   %[[VAL5]] = lit.struct.gep %self[value] : <index> from <!MyValueStruct> loc(#[[MOVE_LOC]])
# CHECK-DAG:   %[[VAL4]] = lit.struct.gep %existing[value] : <index> from <!MyValueStruct> loc(#[[MOVE_LOC]])

# CHECK-DAG: #[[INIT_LOC]] = loc(fused<#[[SP_INIT]]>[#[[DEC_LOC:loc[0-9]*]]])
# CHECK-DAG: #[[COPY_LOC]] = loc(fused<#[[SP_COPY]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[MOVE_LOC]] = loc(fused<#[[SP_MOVE]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[DEC_LOC]] = loc("within split at [[FILENAME]]:

@value
struct MyValueStruct:
    var value: __mlir_type.index

# // -----

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-DAG: ![[SR7:.*]] = !debuginfo.subroutine<(!kgen.pointer<@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22">, index, index) -> (!lit.none): DW_CC_normal>
# CHECK-DAG: #[[SP9:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = "makes_escaping_closure", linkageName = "makes_escaping_closure(,__mlir_type.index,__mlir_type.index)", file = #file, line = [[#LN42:]], scopeLine = [[#SCOPE:]], subprogramFlags = "Definition|Optimized"> : ![[SR7]]

# CHECK-DAG: #[[LOC1:.*]] = loc("{{.*}}":[[#LN42]]:1)
# CHECK-DAG: #[[LOC4:.*]] = loc("{{.*}}":[[#LN42+1]]:3)
# CHECK-DAG: #[[LOC6:.*]] = loc("{{.*}}":[[#LN42]]:27)
# CHECK-DAG: #[[LOC7:.*]] = loc("{{.*}}":[[#LN42]]:50)
# CHECK-DAG: #[[LOC8:.*]] = loc("{{.*}}":[[#LN42+3]]:10)
# CHECK-DAG: #[[LOC9:.*]] = loc("{{.*}}":[[#LN42+3]]:3)

# CHECK-DAG:    lit.func @"makes_escaping_closure
# CHECK-DAG:    debuginfo.value #local_variable1 = %m : index loc(#[[LOC26:.*]])
# CHECK-DAG:    debuginfo.value #local_variable2 = %z : index loc(#[[LOC27:.*]])
# CHECK-DAG:    %anonymous2A = lit.varlet.decl "anonymous*" var synth : <@"${{.*}}"::@"_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22"> loc(#[[LOC28:.*]])
# CHECK-DAG:    %0 = kgen.call @"${{.*}}"::@"_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22"::@"__init__(${{.*}}::_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22=&,__mlir_type.index)"(%anonymous2A, %m) : ("self": !kgen.pointer<@"${{.*}}"::@"_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22"> init_self, "field0": index) -> !lit.none loc(#[[LOC28]])
# CHECK-DAG:    %anonymous2A_0 = lit.varlet.decl "anonymous*" var synth : <@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> loc(#[[LOC28]])
# CHECK-DAG:    %1 = kgen.call @"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"::@"__init__(${{.*}}::_CW_${{.*}}_\22(__mlir_type.index)\22=&,${{.*}}::_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22)"(%anonymous2A_0, %anonymous2A) : ("self": !kgen.pointer<@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> init_self, "impl": !kgen.pointer<@"${{.*}}"::@"_CI_${{.*}}_\22(__mlir_type.index,__mlir_type.index)\22"> borrow_in_mem) -> !lit.none loc(#[[LOC28]])
# CHECK-DAG:    %2 = kgen.call @"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"::@"__copyinit__(${{.*}}::_CW_${{.*}}_\22(__mlir_type.index)\22=&,${{.*}}::_CW_${{.*}}_\22(__mlir_type.index)\22)"(%__result__, %anonymous2A_0) : ("self": !kgen.pointer<@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> init_self, "existing": !kgen.pointer<@"${{.*}}"::@"_CW_${{.*}}_\22(__mlir_type.index)\22"> borrow_in_mem) -> !lit.none loc(#[[LOC29:.*]])

# CHECK-DAG: #[[LOC26]] = loc(fused<#[[SP9]]>[#[[LOC6]]])
# CHECK-DAG: #[[LOC27]] = loc(fused<#[[SP9]]>[#[[LOC7]]])
# CHECK-DAG: #[[LOC28]] = loc(fused<#[[SP9]]>[#[[LOC4]]])
# CHECK-DAG: #[[LOC29]] = loc(fused<#[[SP9]]>[#[[LOC8]]])

fn makes_escaping_closure(m:  __mlir_type.index, z: __mlir_type.index) -> fn(n:__mlir_type.index) escaping ->  __mlir_type.index:
  fn myclosure(n: __mlir_type.index) escaping ->  __mlir_type.index:
      return m
  return myclosure

