// RUN: kgen -target-accelerator amdgpu:94 %s -elaborate -S -o - | FileCheck %s

// CHECK: "amdgpu:94"
kgen.generator export @main() -> !kgen.string {
  %0 = kgen.param.constant : !kgen.string = <#kgen.param.expr<accelerator_arch>>
  hlcf.return %0 : !kgen.string
}
