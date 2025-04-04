// RUN: kgen-opt %s -verify-parameters -lower-lit -verify-parameters -kgen-print-inline-type-values | FileCheck %s

lit.struct.decl @Coro<T: type> register_passable {
  lit.struct.field coro : !kgen.struct<(T)>
}

// CHECK-LABEL: kgen.generator @get_promise
// CHECK-SAME: %arg0: !kgen.struct<(T)>
kgen.generator @get_promise<T: type>(%arg0: !lit.struct<@Coro<:type T>>) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @get_coro
kgen.generator @get_coro<T: type>(%arg0: !lit.struct<@Coro<:type T>>) {
  // CHECK-NEXT: call @get_promise<:type T>(%arg0) : (!kgen.struct<(T)>)
  kgen.call @get_promise<:type T>(%arg0) : (!lit.struct<@Coro<:type T>>) -> ()
  kgen.unreachable
}

lit.struct.decl @Bar<size, dt: dtype> register_passable {
  lit.struct.field value: !pop.simd<size, dt>
}

// CHECK: kgen.struct.generator @[[STRUCT_BAR:.+]]<size, dt: dtype> = struct_inst<"Bar"[size, dt]<size, :dtype dt>(value: simd<size, dt>)>

// CHECK: kgen.generator @anystructs
kgen.generator @anystructs() {
  // COM: Partially bound types will not have uses at the KGEN level.
  // CHECK-NEXT: declare Partial: type = <[typevalue<#kgen.genref<@Bar<?, :dtype f32>>>, simd<?, f32>]>
  kgen.param.declare Partial: meta<!lit.struct<@Bar<?, :dtype f32>, <index>>> = <#lit.bind_type<:meta<!lit.struct<@Bar<?, :dtype ?>, <index, dtype>>> ?, [?, f32]>>

  // CHECK-NEXT: declare BoundFromPartial: type = <[typevalue<#kgen.genref<@[[STRUCT_BAR]]<16, :dtype f32>>>, simd<16, f32>]>
  kgen.param.declare BoundFromPartial: meta<!lit.struct<@Bar<16, :dtype f32>>> = <#lit.bind_type<:meta<!lit.struct<@Bar<?, :dtype f32>, <index>>> Partial, [16]>>
  kgen.return
}
