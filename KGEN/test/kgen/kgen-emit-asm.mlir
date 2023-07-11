// RUN: kgen %s -emit-asm | FileCheck %s

// Check that we generate some ASM properly.
// CHECK: exp_f32

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
