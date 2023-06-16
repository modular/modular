// RUN: kgen-elaborate-opt -elaborate-generators %s | FileCheck %s

kgen.generator @generic_offset_load_store<type: type>(%i: index, %p: !pop.pointer<type>) {
  %0 = pop.offset %p[%i] : !pop.pointer<type>
  %1 = pop.load %0 : !pop.pointer<type>
  pop.store %1, %p : !pop.pointer<type>
  kgen.return
}

// CHECK-LABEL: @"generic_offset_load_store,type=!pop.simd<4, f32>"
// CHECK: pop.offset %{{.*}} : !pop.pointer<simd<4, f32>>

// CHECK-LABEL: @"generic_offset_load_store,type=!pop.scalar<si32>"
// CHECK: pop.offset %{{.*}} : !pop.pointer<scalar<si32>>

kgen.generator @impl(
    %i: index,
    %p0: !pop.pointer<simd<4, f32>>,
    %p1: !pop.pointer<scalar<si32>>) {
  kgen.call @generic_offset_load_store<:type !pop.simd<4, f32>>(%i, %p0) : (index, !pop.pointer<simd<4, f32>>) -> ()
  kgen.call @generic_offset_load_store<:type !pop.scalar<si32>>(%i, %p1) : (index, !pop.pointer<scalar<si32>>) -> ()
  kgen.return
}
