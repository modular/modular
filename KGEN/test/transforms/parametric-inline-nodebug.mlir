// RUN: kgen-opt %s -always-inline-param=nodebug-only=true | FileCheck %s

kgen.generator @nodebug_inline_me() always_inline_no_debug {
  kgen.param.constant = <1>
  kgen.return
}

kgen.generator @always_inline() always_inline {
  kgen.return
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK-NEXT: <1>
  kgen.call @nodebug_inline_me() : () -> ()
  // CHECK-NEXT: call @always_inline
  kgen.call @always_inline() : () -> ()
  kgen.return
}
