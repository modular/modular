// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -allow-unregistered-dialect -emit-bytecode %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #interp.memref<[(my_blob, heap, []), (string_blob, heap, [])], 0, 24> : memref<2xi32>
"some.op"() {a = #interp.memref<[(my_blob, heap, []), (string_blob, heap, [])], 0, 24> : memref<2xi32>} : () -> ()

{-#
  dialect_resources: {
    interp: {
      // CHECK: my_blob: "0x1000
      my_blob: "0x10000000FFFEFDFC",
      // CHECK: string_blob: "hello world"
      string_blob: "hello world"
    }
  }
#-}
