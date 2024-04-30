// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -allow-unregistered-dialect -emit-bytecode %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #interp.memref<[(my_blob, heap, [(1, 1, 3)]), (string_blob, heap, [])], 0, 24> : memref<2xi32>
"some.op"() {a = #interp.memref<[(my_blob, heap, [(1, 1, 3)]), (string_blob, heap, [])], 0, 24> : memref<2xi32>} : () -> ()

// CHECK: #interp.memref<[(variadic, persistent, [])], 0, 0> : memref<1xi32>
"some.op"() {a = #interp.memref<[(variadic, persistent, [])], 0, 0> : memref<1xi32>} : () -> ()

// CHECK: #interp.symbolic_pointer<3> : memref<1xi32>
"some.op"() {a = #interp.symbolic_pointer<3> : memref<1xi32>} : () -> ()

{-#
  dialect_resources: {
    interp: {
      // CHECK: my_blob: "0x1000
      my_blob: "0x10000000FFFEFDFC",
      // CHECK: string_blob: "hello world"
      string_blob: "hello world",
      // CHECK: variadic: "0x0800
      variadic: "0x08000000DEAD"
    }
  }
#-}
