// RUN: kgen-opt -elaborate-generators %s | FileCheck %s

kgen.generator @generic_offset_load_store<type: type>(%i: index, %p: !meta.pointer<type>) {
  %0 = pop.offset %p[%i] : !meta.pointer<type>
  %1 = pop.load %0 : !meta.pointer<type>
  pop.store %1, %p : !meta.pointer<type>
  kgen.return
}

// CHECK-LABEL: @"generic_offset_load_store,type=!meta.simd<4, f32>"
// CHECK: pop.offset %{{.*}} : !meta.pointer<!meta.simd<4, f32>>

// CHECK-LABEL: @"generic_offset_load_store,type=!meta.scalar<si32>"
// CHECK: pop.offset %{{.*}} : !meta.pointer<!meta.scalar<si32>>

kgen.generator @impl(
    %i: index,
    %p0: !meta.pointer<!meta.simd<4, f32>>,
    %p1: !meta.pointer<!meta.scalar<si32>>) {
  kgen.call @generic_offset_load_store<type: type = !meta.simd<4, f32>>(%i, %p0) : (index, !meta.pointer<!meta.simd<4, f32>>) -> ()
  kgen.call @generic_offset_load_store<type: type = !meta.scalar<si32>>(%i, %p1) : (index, !meta.pointer<!meta.scalar<si32>>) -> ()
  kgen.return
}
