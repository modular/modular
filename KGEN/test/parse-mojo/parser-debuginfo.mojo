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

# CHECK-DAG: ![[INT_TYPE:.*]] = !debuginfo.unresolved<!kgen.declref<{{.*}}@"$Int"::@Int>>
# CHECK-DAG: ![[SP_TYPE:.*]] = !debuginfo.subroutine<(!kgen.declref<{{.*}}@"$Int"::@Int>, !kgen.declref<{{.*}}@"$Int"::@Int>) -> (!kgen.declref<{{.*}}@"$Int"::@Int>): DW_CC_normal>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "power", linkageName = "power({{.*}}$Int::Int,{{.*}}$Int::Int)", file = #{{.*}}, line = [[LN:[0-9]+]], scopeLine = [[LN]], subprogramFlags = "Definition|Optimized"> : ![[SP_TYPE]]
# CHECK-DAG: #[[LHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "lhs", file = #{{.*}}, line = [[LN]], arg = 1> : ![[INT_TYPE]]
# CHECK-DAG: #[[RHS_VAR:.*]] = #debuginfo.local_variable<scope = #[[SP]], name = "rhs", file = #{{.*}}, line = [[LN]], arg = 2> : ![[INT_TYPE]]

# CHECK-LABEL: lit.func @"power({{.*}}$Int::Int,{{.*}}$Int::Int)"(
fn power(lhs: Int, rhs: Int) -> Int:
    # CHECK: debuginfo.value #[[LHS_VAR]] = %lhs
    # CHECK: debuginfo.value #[[RHS_VAR]] = %rhs
    return lhs


# // -----

# CHECK-DAG: #[[LOCAL_VAR_I:.*]] = #debuginfo.local_variable<scope = #[[FOR_SP:.*]], name = "i", {{.*}}, line = [[LN:[0-9]+]], arg = 1

# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 = %idx0 : index) -> index {
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
# CHECK-DAG: lit.func @"very_unique_func($Builtin::$Int::Int)"(%very_unique_arg: !kgen.declref<@"$Builtin"::@"$Int"::@Int> loc("[[FILENAME]]"
# CHECK-DAG: debuginfo.value #[[LOCAL_VAR]] = %very_unique_arg : !kgen.declref<@"$Builtin"::@"$Int"::@Int> loc(#[[VALUE_LOC:loc[0-9]+]])

# CHECK-DAG: #[[LOC]] = loc(fused<#[[FILE]]>[#loc{{[0-9]+}}])
# CHECK-DAG: #[[VALUE_LOC]] = loc(fused<#[[SP]]>[#[[LINE_LOC:loc[0-9]+]]])
# CHECK-DAG: #[[LINE_LOC]] = loc("[[FILENAME]]"

fn caller():
    let y = VeryUniqueStruct.very_unique_func(0)

# COM: need this because otherwise FileCheck cannot separate the DAGs.
# CHECK: #-}

# // -----

# COM: These test the locations in the body of synthesized constructors. Because
# COM: of the indeterministic order in which attributes may be printed, we need
# COM: to use CHECK-DAG and order the statements to ensure unique matchings.

# CHECK-DAG: #[[SP_INIT:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE:file[0-9]*]], name = "__init__", linkageName = "__init__($parser-debuginfo::MyValueStruct=&,__mlir_type.index)",
# CHECK-DAG: #[[SP_COPY:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "__copyinit__", linkageName = "__copyinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)",
# CHECK-DAG: #[[SP_MOVE:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = "__moveinit__", linkageName = "__moveinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)",
# CHECK-DAG: #[[FILE]] = #debuginfo.file<"within split at [[FILENAME:.*parser-debuginfo.mojo]]:{{[0-9]+}} offset " in "/">

# CHECK-DAG: lit.func @"__init__($parser-debuginfo::MyValueStruct=&,__mlir_type.index)"
# CHECK-DAG:   pop.store %value, %[[VAL:.*]] : !pop.pointer<index> loc(#[[INIT_LOC:loc[0-9]*]])
# CHECK-DAG:   %[[VAL]] = lit.struct.gep %self[value] : <index> from <@"$parser-debuginfo"::@MyValueStruct> loc(#[[INIT_LOC]])

# CHECK-DAG: lit.func @"__copyinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)"
# CHECK-DAG:   %[[VAL2:.*]] = pop.load %[[VAL1:.*]] : !pop.pointer<index> loc(#[[COPY_LOC:loc[0-9]*]])
# CHECK-DAG:   pop.store %[[VAL2]], %[[VAL0:.*]] : !pop.pointer<index> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL0]] = lit.struct.gep %self[value] : <index> from <@"$parser-debuginfo"::@MyValueStruct> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL1]] = lit.struct.gep %existing[value] : <index> from <@"$parser-debuginfo"::@MyValueStruct> loc(#[[COPY_LOC]])

# CHECK-DAG: lit.func @"__moveinit__($parser-debuginfo::MyValueStruct=&,$parser-debuginfo::MyValueStruct)"
# CHECK-DAG:   %[[VAL3:.*]] = lit.load.consume %[[VAL4:.*]] : !pop.pointer<index> loc(#[[MOVE_LOC:loc[0-9]*]])
# CHECK-DAG:   pop.store %[[VAL3]], %[[VAL5:.*]] : !pop.pointer<index> loc(#[[MOVE_LOC]])
# CHECK-DAG:   %[[VAL5]] = lit.struct.gep %self[value] : <index> from <@"$parser-debuginfo"::@MyValueStruct> loc(#[[MOVE_LOC]])
# CHECK-DAG:   %[[VAL4]] = lit.struct.gep %existing[value] : <index> from <@"$parser-debuginfo"::@MyValueStruct> loc(#[[MOVE_LOC]])

# CHECK-DAG: #[[INIT_LOC]] = loc(fused<#[[SP_INIT]]>[#[[DEC_LOC:loc[0-9]*]]])
# CHECK-DAG: #[[COPY_LOC]] = loc(fused<#[[SP_COPY]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[MOVE_LOC]] = loc(fused<#[[SP_MOVE]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[DEC_LOC]] = loc("within split at [[FILENAME]]:

@value
struct MyValueStruct:
    var value: __mlir_type.index

# // -----

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-DAG: ![[SR:.*]] = !debuginfo.subroutine<(index, index) -> (!kgen.signature<(index borrow) capturing -> index>): DW_CC_normal>
# CHECK-DAG: ![[SR1:.*]] = !debuginfo.subroutine<(index) -> (index): DW_CC_normal>

# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "makes_escaping_closure", linkageName = "makes_escaping_closure(__mlir_type.index,__mlir_type.index)", file = #{{.*}}, line = [[LN_PARENT:[0-9]+]], scopeLine = [[SCOPE_PARENT:.*]], subprogramFlags = "Definition|Optimized"> : ![[SR]]
# CHECK-DAG: #[[SP1:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = "myclosure", linkageName = "myclosure(__mlir_type.index)", file = #{{.*}}, line = [[LN_NESTED:[0-9]+]], scopeLine = [[SCOPE_NESTED:.*]], subprogramFlags = "Definition|Optimized"> : ![[SR1]]

# CHECK-DAG:    lit.func @"makes_escaping_closure
# CHECK-DAG:      %[[V0:.*]] = pop.stack_allocation 1 x index  loc(#[[PARENT_FUNC_LOC0:.*]])
# CHECK-DAG:      pop.store %m, %[[V0]] : !pop.pointer<index> loc(#[[PARENT_FUNC_LOC1:.*]])
# CHECK-DAG:      %[[V1:.*]] = pop.load %[[V0]] : !pop.pointer<index> loc(#[[PARENT_FUNC_LOC0]])
# CHECK-DAG:      lit.func *"myclosure
# CHECK-DAG:        %[[W0:.*]] = pop.stack_allocation 1 x index  loc(#[[NESTED_FUNC_LOC:.*]])
# CHECK-DAG:        pop.store %[[V1]], %[[W0]] : !pop.pointer<index> loc(#[[NESTED_FUNC_LOC]])
# CHECK-DAG:        %[[W4:.*]] = pop.load %[[W0]] : !pop.pointer<index>
# CHECK-DAG:        lit.return %[[W4]] : index
# CHECK-DAG:        lit.end_func loc(#[[NESTED_FUNC_LOC]])
# CHECK-DAG:      } loc(#[[NESTED_FUNC_LOC]])
# CHECK-DAG:      %[[V2:.*]] = kgen.create_closure [<>(index borrow) capturing -> index: *"myclosure(__mlir_type.index)"]()  loc(#[[PARENT_FUNC_LOC0]])
# CHECK-DAG:      lit.return %[[V2]]
# CHECK-DAG:      lit.end_func loc(#[[PARENT_FUNC_LOC0]])
# CHECK-DAG:    } loc(#[[PARENT_FUNC_LOC0]])

# CHECK-DAG: #[[LOC2:.*]] = loc("{{.*}}":[[LN_PARENT]]:1)
# CHECK-DAG: #[[LOC5:.*]] = loc("{{.*}}":[[LN_NESTED]]:3)
# CHECK-DAG: #[[PARENT_FUNC_LOC0]] = loc(fused<#[[SP]]>[#[[LOC2]]
# CHECK-DAG: #[[NESTED_FUNC_LOC]] = loc(fused<#[[SP1]]>[#[[LOC5]]

fn makes_escaping_closure(m:  __mlir_type.index, z: __mlir_type.index) -> fn( __mlir_type.index) escaping ->  __mlir_type.index:
  fn myclosure(n: __mlir_type.index) ->  __mlir_type.index:
      return m
  return myclosure
