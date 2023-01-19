// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func internal @extract_size(%arg0: !llvm.struct<(ptr<i8>, i64)>) -> i64 {
kgen.func @extract_size(%a: !kgen.string) ->  index {
  // CHECK: llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<i8>, i64)>
  %1 = pop.string.size %a
  kgen.return %1: index
}

// CHECK-LABEL: llvm.func internal @extract_addr(%arg0: !llvm.struct<(ptr<i8>, i64)>) -> !llvm.ptr<i8> {
kgen.func @extract_addr(%a: !kgen.string) -> !pop.pointer<scalar<si8>> {
  // CHECK: llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<i8>, i64)>
  %1 = pop.string.address %a
  kgen.return %1: !pop.pointer<scalar<si8>>
}
