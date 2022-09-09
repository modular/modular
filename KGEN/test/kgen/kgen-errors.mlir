// RUN: kgen %s -execute -kernel="exp_f32_kernel:f32():%t_exp_f32_bad.o" -kernel="exp_f32_kernel:f32(f32):%t_exp_f32_good.o" -kernel="badkernel:f32():%t_badkernel.o" -ignore-failure -I %S/../kernels -verify-diagnostics

// expected-error@-3 {{could not find func '@badkernel'}}

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>

// expected-error@below {{command-line specified signature does not match the IR signature}}
kgen.generator @exp_f32(%arg0: f32) -> f32 {
  %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}
