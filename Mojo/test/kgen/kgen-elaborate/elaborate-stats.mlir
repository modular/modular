// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false print-stats=true" -allow-unregistered-dialect -o /dev/null 2>&1 | FileCheck %s

// The report groups the loop's instantiations: one line for the two calls of
// `sum_from_zero`, with their iteration range, the body's size, and the ops
// the unrolled iterations hold once elaborated. The parametric interpreter
// crashes on this loop shape before any report, as it does in
// elaborate-errors.mlir, so only the concrete one runs here.

// CHECK: ===--- Elaboration statistics (--elaboration-stats): host
// CHECK-NEXT: comptime for: 1 loops, 2 instantiations, {{[0-9]+}} ops produced (after elaboration, before inlining)
// CHECK-EMPTY:
// CHECK-NEXT: ops produced  instances   iterations  body ops  loop
// CHECK-NEXT: {{[0-9]+}} 2 2-4 3 {{.*}}elaborate-stats.mlir:[[@LINE+4]]:{{[0-9]+}} @sum_from_zero

kgen.generator @sum_from_zero<upper>() -> index {
  %idx0 = index.constant 0
  %0 = hlcf.comptime.for i in upper
    has_next :(index) -> i1 @count_to_zero_has_next
    get_next_iter :(!kgen.pointer<index> imm_mem, !kgen.pointer<index> byref_result) -> !kgen.none @count_to_zero
    (%arg0 = %idx0 : index) -> index {
    %i = kgen.param.constant = <i>
    %sum = index.add %arg0, %i
    hlcf.comptime.for.continue %sum : index
  } else {
    hlcf.unreachable
  }
  hlcf.return %0 : index
}

kgen.generator export @param_for() {
  kgen.call @sum_from_zero<4>() : () -> index
  kgen.call @sum_from_zero<2>() : () -> index
  hlcf.return
}

kgen.generator @count_to_zero(%arg0: !kgen.pointer<index> imm_mem, %arg1: !kgen.pointer<index> byref_result) -> !kgen.none {
  %i0 = pop.load %arg0 : !kgen.pointer<index>
  %idx1 = index.constant 1
  %1 = index.sub %i0, %idx1
  pop.store %1, %arg1 : !kgen.pointer<index>
  %none = kgen.param.constant: none = <#kgen.none>
  hlcf.return %none : !kgen.none
}

kgen.generator @count_to_zero_has_next(%arg0: index) -> !kgen.scalar<bool> {
  %idx0 = index.constant 0
  %0 = index.cmp ne(%idx0, %arg0)
  %1 = pop.cast_from_builtin %0 : i1 to !kgen.scalar<bool>
  hlcf.return %1 : !kgen.scalar<bool>
}
