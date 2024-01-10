// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @genericSugar<scalar: regtype, T: regtype>
// CHECK-SAME: %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
kgen.generator @genericSugar<scalar: regtype, T: regtype>(
  %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}

// CHECK-LABEL: @memory_only_struct
// CHECK-SAME: %arg0: !kgen.struct<()>,
// CHECK-SAME: %arg1: !kgen.struct<() memoryOnly>,
// CHECK-SAME: %arg2: !kgen.struct<(index, index)>,
// CHECK-SAME: %arg3: !kgen.struct<(index, index) memoryOnly>,
// CHECK-SAME: %arg4: !kgen.struct<((index, index) -> index)>,
// CHECK-SAME: %arg5: !kgen.struct<((index, index) -> index) memoryOnly>
kgen.func @memory_only_struct(
  %arg0: !kgen.struct<()>,
  %arg1: !kgen.struct<() memoryOnly>,
  %arg2: !kgen.struct<(index, index)>,
  %arg3: !kgen.struct<(index, index) memoryOnly>,
  %arg4: !kgen.struct<((index, index) -> index)>,
  %arg5: !kgen.struct<((index, index) -> index) memoryOnly>
) {
  kgen.return
}

// CHECK-LABEL: @type_printing
kgen.generator @type_printing() {
  // CHECK: regtype = <struct<()>>
  kgen.param.declare atype: regtype = <[struct<()>, {}]>
  // CHECK: regtype = <struct<()>>
  kgen.param.declare btype: regtype = <[struct<()>, {}]>
  // CHECK: regtype = <[struct<()>, {"method" : () -> () = @method}]>
  kgen.param.declare btype: regtype = <[struct<()>, {"method" : () -> () = @method}]>
  kgen.return
}

kgen.func @capturing_closure() capturing -> index {
  %idx4 = index.constant 4
  kgen.return %idx4 : index
}
// CHECK-LABEL: @capture_list_round_trip
// CHECK-SAME: %arg0: !kgen.capture_list
kgen.generator @capture_list_round_trip(%arg0: !kgen.capture_list<() capturing -> index : @capturing_closure>) {
  kgen.return
}
