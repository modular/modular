// RUN: kgen-opt -externalize-precompiled-functions %s | FileCheck %s

// CHECK: kgen.link dense_resource<somelib> : tensor<1xui8> as @aLib
kgen.link dense_resource<somelib> : tensor<1xui8> as @aLib

// CHECK: kgen.extern.func @precompiled() -> index from @aLib
kgen.func @precompiled() -> index attributes {precompiledBodyRef = @aLib} {
  %0 = kgen.param.constant = <5>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @main() -> index
kgen.func @main() -> index {
  // CHECK-NEXT: kgen.call @precompiled() : () -> index
  %0 = kgen.call @precompiled() : () -> index
  // CHECK-NEXT: kgen.return {{%[0-9]}} : index
  kgen.return %0 : index
}
