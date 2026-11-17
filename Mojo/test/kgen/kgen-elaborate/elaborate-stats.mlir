// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false print-stats=true" -allow-unregistered-dialect -mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

// The report groups the loop's instantiations: one line for the two calls of
// `sum_from_zero`, with their iteration range, the body's size, and the ops
// the unrolled iterations hold once elaborated. The parametric interpreter
// crashes on this loop shape before any report, as it does in
// elaborate-errors.mlir, so only the concrete one runs here.

// The interpreter tables group the calls by callee, then by operation and
// callee. The loop's iterator calls go through the interpreter result cache.
// The first instantiation interprets `@count_to_zero_has_next` once per
// iteration plus the last check (5 times) and `@count_to_zero` 4 times. The
// second instantiation's 3 + 2 calls all hit the cache. Which instantiation
// fills the cache depends on scheduling, so the run is single-threaded to
// keep the counts exact. The second `kgen.param.apply` of `@fma` hits the
// cache the first filled, and the two share a callee row.

// CHECK: ===--- Elaboration statistics (--elaboration-stats): host
// CHECK-NEXT: comptime for: 1 loops, 2 instantiations, {{[0-9]+}} ops produced (after elaboration, before inlining)
// CHECK-EMPTY:
// CHECK-NEXT: ops produced  instances   iterations  body ops  loop
// CHECK-NEXT: {{[0-9]+}} 2 2-4 3 {{.*}}elaborate-stats.mlir:[[#LOOP:@LINE+18]]:{{[0-9]+}} @sum_from_zero
// CHECK-EMPTY:
// CHECK-NEXT: interpreter: 16 calls, 6 cache hits, 10 interpreted in {{[0-9.]+}} s CPU
// CHECK-EMPTY:
// CHECK-NEXT: By callee, each function's calls summed over all its call sites:
// CHECK-NEXT: CPU s interpreted cache hits max s call sites callee
// CHECK-DAG: {{[0-9.]+}} 5 3 {{[0-9.]+}} 1 @count_to_zero_has_next
// CHECK-DAG: {{[0-9.]+}} 4 2 {{[0-9.]+}} 1 @count_to_zero{{$}}
// CHECK-DAG: {{[0-9.]+}} 1 1 {{[0-9.]+}} 2 @fma
// CHECK: By call site, one operation's calls to one function:
// CHECK-NEXT: CPU s interpreted cache hits max s call site callee
// CHECK-DAG: {{[0-9.]+}} 5 3 {{[0-9.]+}} {{.*}}elaborate-stats.mlir:[[#LOOP]]:{{[0-9]+}} @count_to_zero_has_next
// CHECK-DAG: {{[0-9.]+}} 4 2 {{[0-9.]+}} {{.*}}elaborate-stats.mlir:[[#LOOP]]:{{[0-9]+}} @count_to_zero{{$}}
// CHECK-DAG: {{[0-9.]+}} 1 0 {{[0-9.]+}} {{.*}}elaborate-stats.mlir:[[#FMA:@LINE+47]]:{{[0-9]+}} @fma
// CHECK-DAG: 0.0000 0 1 0.0000 {{.*}}elaborate-stats.mlir:[[#FMA+1]]:{{[0-9]+}} @fma

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

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  hlcf.return %1 : index
}

kgen.generator export @apply_twice() {
  kgen.param.apply x = [(index, index, index) -> index: @fma](1, 2, 3)
  kgen.param.apply y = [(index, index, index) -> index: @fma](1, 2, 3)
  hlcf.return
}
