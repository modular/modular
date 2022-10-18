// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: lit.func @trivial_generator(%name: si32)
lit.func @trivial_generator(%name: si32) -> si32 {
  // CHECK-NEXT: kgen.return %name : si32
  kgen.return %name : si32
}

// CHECK-LABEL: kgen.generator.interface @itf<ty: dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>
kgen.generator.interface @itf<ty : dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @impl1<ty: dtype>(%arg0: !pop.scalar<ty>
// CHECK-NEXT: implements @itf {
lit.func @impl1<ty : dtype>(%arg0: !pop.scalar<ty>) -> !pop.scalar<ty>
  implements @itf {
  kgen.return %arg0 : !pop.scalar<ty>
}

// One implementation of dynamic_thing
// CHECK-LABEL: lit.func @vardecl
// CHECK-NEXT: %x = lit.var.decl "x" : <scalar<ty>>
lit.func @vardecl<ty : dtype>() {
  %x = lit.var.decl "x": !pop.pointer<scalar<ty>>
  kgen.return
}

// CHECK-LABEL: lit.struct.decl @SomeStruct<ty: dtype> {
lit.struct.decl @SomeStruct<ty: dtype> {
  // CHECK-NEXT: lit.func @foo() {
  lit.func @foo() {
    kgen.return
  }

  // CHECK: %size = lit.var.decl "size" : <scalar<ty>>
  %size = lit.var.decl "size" : !pop.pointer<scalar<ty>>
}

%thing = lit.var.decl "thing" : !pop.pointer<!kgen.ref<@Int>>
