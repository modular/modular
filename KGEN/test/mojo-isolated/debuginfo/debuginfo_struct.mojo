# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -O0 -mlir-print-debuginfo %s | FileCheck %s

# CHECK: #[[SOURCENAME_INT:.*]] = #debuginfo.source_name<(struct)"Int" from {{.*}}>
# CHECK-DAG: #[[SOURCENAME_RP:.*]] = #debuginfo.source_name<(struct)"MyRP"[#[[SOURCENAME_INT]]] from <(module)"debuginfo_struct">>

# CHECK-DAG: #[[SOURCENAME_RP3:.*]] = #debuginfo.source_name<(struct)"MyRP"[#[[SOURCENAME_INT]]]<":{{.*}} {3}"> from <(module)"debuginfo_struct">>
# CHECK-DAG: #[[SOURCENAME_DATA:.*]] = #debuginfo.source_name<(struct)"MyData"[#[[SOURCENAME_INT]], #[[SOURCENAME_RP3]], <"type">] from <(module)"debuginfo_struct">>


# CHECK: lit.struct.decl @MyRP
# CHECK-SAME: sourceName = #[[SOURCENAME_RP]]
@fieldwise_init
@register_passable("trivial")
struct MyRP[A: Int]:
    var a: Int
    var b: Int

    @implicit
    fn __init__(out self, b: Int):
        self.a = A
        self.b = b


# CHECK: lit.struct.decl @MyData
# CHECK-SAME: sourceName = #[[SOURCENAME_DATA]]
struct MyData[A: Int, B: MyRP[3], C: AnyTrivialRegType]:
    var a: Int
    var b: MyRP[3]
    var c: C

    @implicit
    fn __init__(out self, c: C):
        self.a = A
        self.b = B
        self.c = c


fn entry():
    alias rp = MyRP[3](4)
    var data = MyData[7, rp, MyRP[3]](rp)
