# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


# CHECK-LABEL: lit.globalvar.decl @x : index
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@x : <index
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <1>
# CHECK-NEXT: lit.ref.store %[[VAL]], %[[REF]]
var x = __mlir_attr.`1 : index`


struct ConvertibleFromInt:
    fn __init__(inout self, v: Int):
        pass


# CHECK-LABEL: lit.globalvar.decl @y : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@y : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <2>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[REF]], %[[VAL]])
let y: ConvertibleFromInt = __mlir_attr.`2 : index`

# CHECK-LABEL: lit.globalvar.decl @z : !ConvertibleFromInt
# CHECK-DAG: %[[REF:.*]] = lit.globalvar.ref {{.*}}@z : <!ConvertibleFromInt
# CHECK-DAG: %[[VAL:.*]] = kgen.param.constant = <3>
# CHECK-NEXT: lit.call {{.*}}@ConvertibleFromInt::@"__init__{{.*}}(%[[REF]], %[[VAL]])
let z = ConvertibleFromInt(__mlir_attr.`3: index`)
