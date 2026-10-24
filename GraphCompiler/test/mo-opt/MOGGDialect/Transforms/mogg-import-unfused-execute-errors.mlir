// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%unfused_execute_invalid_pkg" 2>&1 | FileCheck %s

// Each kernel in the unfused_execute_invalid_stubs package declares a second
// `execute` overload that breaks one rule of an `execute` overload pair. Import
// reports every violation rather than stopping at the first.

// CHECK-DAG: error: kernel 'unfused_not_allowed' declares a second `execute` overload; the only supported second `execute` receives the kernel's outputs unfused when nothing is fused into them, and it is a stopgap that only a specific set of pre-existing kernels may declare{{ *$}}
// CHECK-DAG: error: kernel 'trait_unfused_execute' (type="gpu", api="cuda", arch="sm_100a") declares a second `execute` overload;
// CHECK-DAG: error: kernel 'unfused_invalid_both_fused' declares two `execute` overloads, so exactly one must declare a fused output
// CHECK-DAG: error: kernel 'unfused_invalid_three' declares 3 `execute` overloads; declare a single `execute`{{ *$}}
// CHECK-DAG: error: unfused `execute` overload input at position 1 differs from the fused overload's
// CHECK-DAG: error: unfused `execute` overload takes 1 output(s) but the fused overload takes 2
// CHECK-DAG: error: unfused `execute` overload must raise iff the fused overload does
// CHECK-DAG: error: unfused `execute` overload must declare the same parameters as the fused overload, in the same order
// CHECK-DAG: error: an `execute` overload pair must bind every tensor argument by tensor traits (argument at position 1)

module {}
