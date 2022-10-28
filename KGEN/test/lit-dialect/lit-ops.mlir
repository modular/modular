// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.func @trivial_generator(%name: si32)
lit.func @trivial_generator(%name: si32) -> si32 {
  // CHECK-NEXT: kgen.return %name : si32
  kgen.return %name : si32
}

// CHECK-LABEL: kgen.generator.interface @itf<ty: dtype>(!pop.simd<1, ty>) -> !pop.simd<1, ty>
kgen.generator.interface @itf<ty : dtype>(!pop.simd<1, ty>) -> !pop.simd<1, ty>

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @impl1<ty: dtype>(%arg0: !pop.simd<1, ty>
// CHECK-NEXT: implements @itf {
lit.func @impl1<ty : dtype>(%arg0: !pop.simd<1, ty>) -> !pop.simd<1, ty>
  implements @itf {
  kgen.return %arg0 : !pop.simd<1, ty>
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @vardecl
// CHECK-NEXT: %x = lit.var.decl "x" : <simd<1, ty>>
lit.func @vardecl<ty : dtype>() {
  %x = lit.var.decl "x": !pop.pointer<simd<1, ty>>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype> {
lit.struct.decl @SomeStruct<ty: dtype> {
  // CHECK-NEXT: lit.func @foo() {
  lit.func @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" : <simd<1, ty>>
  %size = lit.var.decl "size" : !pop.pointer<simd<1, ty>>
}

%thing = lit.var.decl "thing" : !pop.pointer<!kgen.ref<@Int>>
