// RUN: kgen -emit %s -o %t.a
// RUN: llvm-objdump %t.a -t

// CHECK-LABEL: (0.o):
// CHECK-DAG: kgen_main
// CHECK-DAG: kgen_use

// CHECK-LABEL: (1.o):
// CHECK-DAG: kgen_split

kgen.func @keep0(%arg0: !pop.pointer<i32>) {
  pop.external_call @extern0(%arg0) : (!pop.pointer<i32>) -> ()
  kgen.return
}

kgen.func @keep1(%arg0: !pop.pointer<i32>) {
  pop.external_call @extern1(%arg0) : (!pop.pointer<i32>) -> ()
  kgen.return
}

kgen.func @ctor() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep0(%0) : (!pop.pointer<i32>) -> ()
  kgen.return
}

kgen.func @dtor() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep0(%0) : (!pop.pointer<i32>) -> ()
  kgen.return
}

kgen.global @global : i32 (2, @ctor, @dtor)

kgen.export @kgen_main
kgen.func @kgen_main() {
  %0 = pop.global.address @global : <i32>
  kgen.call @keep1(%0) : (!pop.pointer<i32>) -> ()
  kgen.return
}

kgen.export @kgen_use
kgen.func @kgen_use() -> !pop.pointer<i32> {
  %0 = pop.global.address @global : <i32>
  kgen.return %0 : !pop.pointer<i32>
}

kgen.func @noop() {
  kgen.return
}

kgen.global @another : i32 (2, @noop, @noop)

kgen.func @split_callee() {
  kgen.return
}

kgen.export @kgen_split
kgen.func @kgen_split() {
  kgen.call @split_callee() : () -> ()
  kgen.return
}
