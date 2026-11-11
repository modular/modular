// RUN: kgen-opt %s -force-always-inline | FileCheck %s

// CHECK: kgen.func @helper() always_inline
kgen.func @helper() {
  hlcf.return
}

// CHECK: kgen.func export @kernel() always_inline
kgen.func export @kernel() {
  hlcf.return
}

// An explicit `@no_inline` is overridden too. The override warning is emitted
// by the pipeline's callback, not the bare pass, so it isn't checked here.
// CHECK: kgen.func @no_inline_fn() always_inline
kgen.func @no_inline_fn() no_inline {
  hlcf.return
}

// An external function has no body to inline, so the pass leaves its inline
// level alone.
// CHECK: kgen.func @external_fn() no_inline {
// CHECK-NEXT: hlcf.unreachable
// CHECK-NEXT: } <external>
kgen.func @external_fn() no_inline {
  hlcf.unreachable
} <external>
