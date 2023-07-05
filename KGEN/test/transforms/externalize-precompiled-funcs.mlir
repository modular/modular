// RUN: kgen-opt -externalize-precompiled-functions %s | FileCheck %s

// CHECK: kgen.link "somelib.a" as @aLib
kgen.link "somelib.a" as @aLib

// CHECK-NOT: kgen.func @precompiled
kgen.func @precompiled() -> index attributes {precompiledBodyRef = @aLib} {
  %0 = kgen.param.constant = <5>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @main() -> index
kgen.func @main() -> index {
  // CHECK-NEXT: pop.external_call @precompiled() from @aLib : () -> index
  %0 = kgen.call @precompiled() : () -> index
  // CHECK-NEXT: kgen.return {{%[0-9]}} : index
  kgen.return %0 : index
}
