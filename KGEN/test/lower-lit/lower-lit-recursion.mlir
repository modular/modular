// RUN: kgen-opt -lower-semantic-cf %s | FileCheck %s

// CHECK-LABEL: lit.func @recurse
// CHECK-SAME (%x: !pop.scalar<index>) -> !pop.scalar<index> {
// CHECK-NEXT: %0 = kgen.call @recurse(%x) : (!pop.scalar<index>) -> !pop.scalar<index>
// CHECK-NEXT: kgen.return %0 : !pop.scalar<index>
// CHECK-NEXT:}
lit.func @recurse(%x: !pop.scalar<index>) -> !pop.scalar<index> {
  %7 = kgen.call @recurse(%x) : (!pop.scalar<index>) -> !pop.scalar<index>
  lit.return %7 : !pop.scalar<index>
  lit.end_func
}
