// RUN: not kgen %s -emit-llvm 2>&1 | FileCheck %s -DOPT=-emit-llvm
// RUN: not kgen %s -emit-llvm=opt 2>&1 | FileCheck %s -DOPT=-emit-llvm=opt
// RUN: not kgen %s -emit-asm 2>&1 | FileCheck %s -DOPT=-emit-asm
// RUN: not kgen %s -emit-asm-verbose 2>&1 | FileCheck %s -DOPT=-emit-asm-verbose
// RUN: not kgen %s -emit-header 2>&1 | FileCheck %s -DOPT=-emit-header

// CHECK: kgen: error: unrecognized argument '[[OPT]]'

kgen.generator export @some_func(%arg0: f32) -> (f32, f32) {
  hlcf.return %arg0, %arg0 : f32, f32
}
