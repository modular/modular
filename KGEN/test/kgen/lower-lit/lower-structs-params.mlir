// RUN: kgen-opt %s -verify-parameters -lower-lit-types -verify-parameters | FileCheck %s

lit.struct.decl @Coro<T: type>  {
  lit.struct.field coro : !pop.coroutine<() -> !kgen.paramref<T>>
}

// CHECK-LABEL: kgen.generator @get_promise
// CHECK-SAME: %arg0: !pop.coroutine<() -> !kgen.paramref<T>>
kgen.generator @get_promise<T: type>(%arg0: !kgen.declref<@Coro<T: type = T>>) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @get_coro
kgen.generator @get_coro<T: type>(%arg0: !kgen.declref<@Coro<T: type = T>>) {
  // CHECK-NEXT: call @get_promise<:type T>(%arg0) : (!pop.coroutine<() -> !kgen.paramref<T>>)
  kgen.call @get_promise<:type T>(%arg0) : (!kgen.declref<@Coro<T: type = T>>) -> ()
  kgen.unreachable
}
