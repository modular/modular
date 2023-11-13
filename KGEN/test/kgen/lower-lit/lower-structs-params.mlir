// RUN: kgen-opt %s -verify-parameters -lower-lit-types -verify-parameters | FileCheck %s

lit.struct.decl @Coro<T: regtype> register_passable {
  lit.struct.field coro : !pop.coroutine<() -> !kgen.paramref<T>>
}

// CHECK-LABEL: kgen.generator @get_promise
// CHECK-SAME: %arg0: !pop.coroutine<() -> !kgen.paramref<T>>
kgen.generator @get_promise<T: regtype>(%arg0: !kgen.declref<@Coro<:regtype T>>) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @get_coro
kgen.generator @get_coro<T: regtype>(%arg0: !kgen.declref<@Coro<:regtype T>>) {
  // CHECK-NEXT: call @get_promise<:regtype T>(%arg0) : (!pop.coroutine<() -> !kgen.paramref<T>>)
  kgen.call @get_promise<:regtype T>(%arg0) : (!kgen.declref<@Coro<:regtype T>>) -> ()
  kgen.unreachable
}

lit.struct.decl @Bar<size, dt: dtype> register_passable {
  lit.struct.field value: !pop.simd<size, dt>
}

// CHECK-LABEL: kgen.generator @metatypes
kgen.generator @metatypes() {
  // COM: Partially bound types will not have uses at the KGEN level.
  // CHECK-NEXT: declare Partial: regtype = <simd<?, f32>>
  kgen.param.declare Partial: metatype<@Bar<?, :dtype f32>, <index>> = <#lit.bind_type<:metatype<@Bar<?, :dtype ?>, <index, dtype>> ?, [?, f32]>>

  // CHECK-NEXT: declare BoundFromPartial: regtype = <simd<16, f32>>
  kgen.param.declare BoundFromPartial: metatype<@Bar<16, :dtype f32>> = <#lit.bind_type<:metatype<@Bar<?, :dtype f32>, <index>> Partial, [16]>>
  kgen.return
}
