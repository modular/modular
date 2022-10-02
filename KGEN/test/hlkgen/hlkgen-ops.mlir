// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: hlkgen.generator @trivial_generator
// CHECK-SAME: %[[ARG0:.*]]: si32
hlkgen.generator @trivial_generator(%arg0: si32) -> si32 {
  // CHECK-NEXT: kgen.return %[[ARG0]] : si32
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator.interface @itf<ty: dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>
kgen.generator.interface @itf<ty : dtype>(!pop.scalar<ty>) -> !pop.scalar<ty>

// One implementation of dynamic_thing
// CHECK-LABEL: hlkgen.generator @impl1<ty: dtype>
// CHECK-SAME: %[[ARG0:.*]]: !pop.scalar<ty>
// CHECK-NEXT: implements @itf {
hlkgen.generator @impl1<ty : dtype>(%arg0: !pop.scalar<ty>) -> !pop.scalar<ty>
  implements @itf {
  kgen.return %arg0 : !pop.scalar<ty>
}

// One implementation of dynamic_thing
// CHECK-LABEL: hlkgen.generator @vardecl
// CHECK-NEXT: %0 = hlkgen.var.decl "x": <!pop.scalar<ty>>
hlkgen.generator @vardecl<ty : dtype>() {
  %0 = hlkgen.var.decl "x": !pop.pointer<!pop.scalar<ty>>
  kgen.return
}
