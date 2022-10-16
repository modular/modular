// RUN: kgen-opt -lower-zap-to-pop -lower-kgen-to-llvm -pass-pipeline='kgen.func(lower-pop-to-llvm)' -lower-to-llvm %s | kgen-translate -mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @naive_matmul
kgen.func public @naive_matmul(%a: !zap.ndbuffer<[?, ?], f32>,
                        %b: !zap.ndbuffer<[?, ?], f32>,
                        %c: !zap.ndbuffer<[?, ?], f32>) {
  %zero = index.constant 0
  %one = index.constant 1

  %M = zap.ndbuffer.dim %a[0] : !zap.ndbuffer<[?, ?], f32>
  %N = zap.ndbuffer.dim %b[1] : !zap.ndbuffer<[?, ?], f32>
  %K = zap.ndbuffer.dim %a[1] : !zap.ndbuffer<[?, ?], f32>

  %BK = zap.ndbuffer.dim %b[0] : !zap.ndbuffer<[?, ?], f32>
  %K_eq_BK0 = index.cmp eq(%K, %BK)
  %K_eq_BK = pop.cast_from_builtin %K_eq_BK0 : i1 to !pop.scalar<bool>
  zap.debug_assert %K_eq_BK, "K != BK" : !pop.scalar<bool>

  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %init = pop.constant(0.0 : f32) : !pop.scalar<f32>
      %cij = scf.for %k = %zero to %K step %one
                    iter_args(%sum = %init) -> (!pop.scalar<f32>) {
        %aik = zap.ndbuffer.load %a[%i, %k] : !zap.ndbuffer<[?, ?], f32>
        %bkj = zap.ndbuffer.load %b[%k, %j] : !zap.ndbuffer<[?, ?], f32>
        %aikbkj = pop.mul %aik, %bkj : !pop.scalar<f32>
        %cij = pop.add %sum, %aikbkj : !pop.scalar<f32>
        scf.yield %cij : !pop.scalar<f32>
      }
      zap.ndbuffer.store %cij, %c[%i, %j] : !zap.ndbuffer<[?, ?], f32>
    }
  }

  kgen.return
}
