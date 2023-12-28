# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | FileCheck %s


# COM: These test the locations in the body of synthesized constructors. Because
# COM: of the indeterministic order in which attributes may be printed, we need
# COM: to use CHECK-DAG and order the statements to ensure unique matchings.

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

# CHECK-DAG: #__init___name = #debuginfo.source_name<(fn)"__init__"(#MyValueStruct_name, <"index">) from #MyValueStruct_name>
# CHECK-DAG: #__moveinit___name1 = #debuginfo.source_name<(fn)"__moveinit__"(#MyValueStruct_name, #MyValueStruct_name) from #MyValueStruct_name>
# CHECK-DAG: #[[SP_INIT:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE:file[0-9]*]], name = #__init___name, linkageName = "__init__(${{.*}}::MyValueStruct=&,__mlir_type.index)",
# CHECK-DAG: #[[SP_COPY:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = #__copyinit___name1, linkageName = "__copyinit__(${{.*}}::MyValueStruct=&,${{.*}}::MyValueStruct)",
# CHECK-DAG: #[[SP_MOVE:subprogram[0-9]*]] = #debuginfo.subprogram<compileUnit = {{#compile_unit[0-9]*}}, scope = #[[FILE]], name = #__moveinit___name1, linkageName = "__moveinit__(${{.*}}::MyValueStruct=&,${{.*}}::MyValueStruct)",
# CHECK-DAG: #[[FILE]] = #debuginfo.file<"[[FILENAME:.*debuginfo-synth-ctor.mojo]]" in "/">

# CHECK-DAG: lit.func @"__init__(${{.*}}::MyValueStruct=&,__mlir_type.index)"(%[[SELF:.*]][*""]:
# CHECK-DAG:   pop.store %value, %[[VAL:.*]] : !kgen.pointer<index> loc(#[[INIT_LOC:loc[0-9]*]])
# CHECK-DAG:   %[[VAL]] = lit.struct.gep %[[SELF]][value] : <index> from <!MyValueStruct> loc(#[[INIT_LOC]])

# CHECK-DAG: lit.func @"__copyinit__(${{.*}}::MyValueStruct=&,${{.*}}::MyValueStruct)"
# CHECK-DAG:   %[[VAL2:.*]] = pop.load %[[VAL1:.*]] : !kgen.pointer<index> loc(#[[COPY_LOC:loc[0-9]*]])
# CHECK-DAG:   pop.store %[[VAL2]], %[[VAL0:.*]] : !kgen.pointer<index> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL0]] = lit.struct.gep %self[value] : <index> from <!MyValueStruct> loc(#[[COPY_LOC]])
# CHECK-DAG:   %[[VAL1]] = lit.struct.gep %other[value] : <index> from <!MyValueStruct> loc(#[[COPY_LOC]])

# CHECK-DAG: lit.func @"__moveinit__{{.*}}::MyValueStruct=&,${{.*}}::MyValueStruct)"
# CHECK-DAG:   [[VAL5:%.*]] = lit.struct.gep %self[value] : <index> from <!MyValueStruct> loc(#[[MOVE_LOC:loc[0-9]*]])
# CHECK-DAG:   [[VAL4:%.*]] = lit.struct.gep %other[value]
# CHECK-DAG:   %2 = builtin.unrealized_conversion_cast [[VAL4]]
# CHECK-DAG:   [[VAL3:%.*]] = lit.load.consume %2
# CHECK-DAG:   pop.store [[VAL3]], [[VAL5:.*]] : !kgen.pointer<index> loc(#[[MOVE_LOC]])

# CHECK-DAG: #[[INIT_LOC]] = loc(fused<#[[SP_INIT]]>[#[[DEC_LOC:loc[0-9]*]]])
# CHECK-DAG: #[[COPY_LOC]] = loc(fused<#[[SP_COPY]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[MOVE_LOC]] = loc(fused<#[[SP_MOVE]]>[#[[DEC_LOC]]])
# CHECK-DAG: #[[DEC_LOC]] = loc("[[FILENAME]]


@value
struct MyValueStruct:
    var value: __mlir_type.index
