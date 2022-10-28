// RUN: kgen-execute %s -execute -func="use_struct:f32()" | FileCheck %s

kgen.struct.decl @FooStruct<T:type> {
  value : !kgen.paramref<T>
}

kgen.func public @use_struct() -> f32 {
  %0 = pop.constant(1.0 : f32) : !pop.simd<1, f32>
  %1 = pop.cast_to_builtin %0 : !pop.simd<1, f32> to f32
  %2 = kgen.struct.create(%1) : (f32) -> !kgen.ref<@FooStruct<T:type = f32>>
  %3 = kgen.struct.extract %2[value] : f32 from !kgen.ref<@FooStruct<T:type = f32>>
  kgen.return %3 : f32
}

// CHECK: 'use_struct' returned 1.0
