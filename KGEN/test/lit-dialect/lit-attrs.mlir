// RUN: kgen-opt %s -allow-unregistered-dialect | FileCheck %s

// CHECK: #lit.none : i32
"a"() {a = #lit.none : i32} : () -> ()
