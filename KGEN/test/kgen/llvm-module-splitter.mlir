// RUN: kgen -emit %s -o %t.a
// RUN: llvm-objdump %t.a -t

// CHECK-LABEL: (0.o):
// CHECK-DAG: kgen_main
// CHECK-DAG: kgen_use
// CHECK-DAG: kgen_global_ctor
// CHECK-DAG: kgen_global_dtor

// CHECK-LABEL: (1.o):
// CHECK-DAG: kgen_split

kgen.func @keep0(%arg0: !kgen.pointer<i32>) {
  pop.external_call @extern0(%arg0) : (!kgen.pointer<i32>) -> ()
  kgen.return
}

kgen.func @keep1(%arg0: !kgen.pointer<i32>) {
  pop.external_call @extern1(%arg0) : (!kgen.pointer<i32>) -> ()
  kgen.return
}

kgen.func @kgen_global_ctor() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep0(%0) : (!kgen.pointer<i32>) -> ()
  kgen.return
}

kgen.func @kgen_global_dtor() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep0(%0) : (!kgen.pointer<i32>) -> ()
  kgen.return
}

kgen.global @global : i32 [@kgen_global_ctor, @kgen_global_dtor](2)

kgen.func export @kgen_main() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep1(%0) : (!kgen.pointer<i32>) -> ()
  kgen.return
}

kgen.func export @kgen_use() -> !kgen.pointer<i32> {
  %0 = pop.global.address @global : <i32>
  kgen.return %0 : !kgen.pointer<i32>
}

kgen.func @noop() {
  kgen.return
}

kgen.global @another : i32 [@noop, @noop](2)

kgen.func @split_callee() {
  kgen.return
}

kgen.func export @kgen_split() {
  kgen.call @split_callee() : () -> ()
  kgen.return
}
