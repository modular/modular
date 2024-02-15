// RUN: kgen-opt %s -verify-parameters -lower-lit-types -verify-parameters | FileCheck %s

lit.struct.decl @Coro<T: type> register_passable {
  lit.struct.field coro : !pop.coroutine<() -> !kgen.paramref<T>>
}

// CHECK-LABEL: kgen.generator @get_promise
// CHECK-SAME: %arg0: !pop.coroutine<() -> !kgen.paramref<T>>
kgen.generator @get_promise<T: type>(%arg0: !lit.declref<@Coro<:type T>>) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @get_coro
kgen.generator @get_coro<T: type>(%arg0: !lit.declref<@Coro<:type T>>) {
  // CHECK-NEXT: call @get_promise<:type T>(%arg0) : (!pop.coroutine<() -> !kgen.paramref<T>>)
  kgen.call @get_promise<:type T>(%arg0) : (!lit.declref<@Coro<:type T>>) -> ()
  kgen.unreachable
}

lit.struct.decl @Bar<size, dt: dtype> register_passable {
  lit.struct.field value: !pop.simd<size, dt>
}

// CHECK-LABEL: kgen.generator @metatypes
kgen.generator @metatypes() {
  // COM: Partially bound types will not have uses at the KGEN level.
  // CHECK-NEXT: declare Partial: type = <simd<?, f32>>
  kgen.param.declare Partial: metatype<@Bar<?, :dtype f32>, <index>> = <#lit.bind_type<:metatype<@Bar<?, :dtype ?>, <index, dtype>> ?, [?, f32]>>

  // CHECK-NEXT: declare BoundFromPartial: type = <simd<16, f32>>
  kgen.param.declare BoundFromPartial: metatype<@Bar<16, :dtype f32>> = <#lit.bind_type<:metatype<@Bar<?, :dtype f32>, <index>> Partial, [16]>>
  kgen.return
}
