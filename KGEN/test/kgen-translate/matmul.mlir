// RUN: kgen-opt -pass-pipeline='builtin.module(lower-zap-to-pop,lower-to-llvm)' %s | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_bit_width=64, simd_bit_width=128>} {

// CHECK-LABEL: define internal void @naive_matmul
kgen.func @naive_matmul(%a: !zap.ndbuffer<[?, ?], f32>,
                        %b: !zap.ndbuffer<[?, ?], f32>,
                        %c: !zap.ndbuffer<[?, ?], f32>) {
  %zero = index.constant 0
  %one = index.constant 1

  %M = zap.ndbuffer.dim %a[0] : !zap.ndbuffer<[?, ?], f32>
  %N = zap.ndbuffer.dim %b[1] : !zap.ndbuffer<[?, ?], f32>
  %K = zap.ndbuffer.dim %a[1] : !zap.ndbuffer<[?, ?], f32>

  %BK = zap.ndbuffer.dim %b[0] : !zap.ndbuffer<[?, ?], f32>
  %K_eq_BK0 = index.cmp eq(%K, %BK)
  %K_eq_BK = pop.cast_from_builtin %K_eq_BK0 : i1 to !pop.simd<1, bool>

  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %init = kgen.param.constant: scalar<f32> = <<"0.0">>
      %cij = scf.for %k = %zero to %K step %one
                    iter_args(%sum = %init) -> (!pop.simd<1, f32>) {
        %aik = zap.ndbuffer.load %a[%i, %k] : !zap.ndbuffer<[?, ?], f32>, !pop.simd<1, f32>
        %bkj = zap.ndbuffer.load %b[%k, %j] : !zap.ndbuffer<[?, ?], f32>, !pop.simd<1, f32>
        %aikbkj = pop.mul %aik, %bkj : !pop.simd<1, f32>
        %cij = pop.add %sum, %aikbkj : !pop.simd<1, f32>
        scf.yield %cij : !pop.simd<1, f32>
      }
      zap.ndbuffer.store %cij, %c[%i, %j] : !pop.simd<1, f32>, !zap.ndbuffer<[?, ?], f32>
    }
  }

  kgen.return
}

}
