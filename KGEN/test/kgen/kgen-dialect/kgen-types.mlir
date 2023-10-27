// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @sugaredScalar
// CHECK-SAME: %arg0: !kgen.pointer<*"scalar">
kgen.generator @sugaredScalar<scalar: type>(%arg0: !kgen.pointer<*"scalar">) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}

// CHECK-LABEL: @memory_only_struct
// CHECK-SAME: %arg0: !kgen.struct<>
// CHECK-SAME: %arg1: !kgen.struct<>
// CHECK-SAME: %arg2: !kgen.struct<() memoryOnly>
// CHECK-SAME: %arg3: !kgen.struct<index, index>
// CHECK-SAME: %arg4: !kgen.struct<(index, index) memoryOnly>
kgen.func @memory_only_struct(
  %arg0: !kgen.struct<>,
  %arg1: !kgen.struct<()>,
  %arg2: !kgen.struct<() memoryOnly>,
  %arg3: !kgen.struct<(index, index)>,
  %arg4: !kgen.struct<(index, index) memoryOnly>
) {
  kgen.return
}
