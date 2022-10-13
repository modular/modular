// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: naive_matmul
// CHECK: %[[A:.*]]: [[TENSOR_TYPE:.*]], f32>,
// CHECK: %[[B:.*]]: [[TENSOR_TYPE]]
// CHECK: %[[C:.*]]: [[TENSOR_TYPE]]
kgen.func @naive_matmul(%a: !zap.tensor<[?, ?], f32>,
                        %b: !zap.tensor<[?, ?], f32>,
                        %c: !zap.tensor<[?, ?], f32>) {
  %zero = index.constant 0
  %one = index.constant 1

  %M = zap.tensor.dim %a[0] : !zap.tensor<[?, ?], f32>
  %N = zap.tensor.dim %b[1] : !zap.tensor<[?, ?], f32>
  %K = zap.tensor.dim %a[1] : !zap.tensor<[?, ?], f32>

  %BK = zap.tensor.dim %b[0] : !zap.tensor<[?, ?], f32>
  %K_eq_BK0 = index.cmp eq(%K, %BK)
  %K_eq_BK = pop.cast_from_builtin %K_eq_BK0 : i1 to !pop.scalar<bool>
  zap.debug_assert %K_eq_BK, "K != BK" : !pop.scalar<bool>


  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %init = pop.constant(0.0 : f32) : !pop.scalar<f32>
      %cij = scf.for %k = %zero to %K step %one
                    iter_args(%sum = %init) -> (!pop.scalar<f32>) {
        %aik = zap.tensor.load %a[%i, %k] : !zap.tensor<[?, ?], f32>
        %bkj = zap.tensor.load %b[%k, %j] : !zap.tensor<[?, ?], f32>
        %aikbkj = pop.mul %aik, %bkj : !pop.scalar<f32>
        %cij = pop.add %sum, %aikbkj : !pop.scalar<f32>
        scf.yield %cij : !pop.scalar<f32>
      }
      zap.tensor.store %cij, %c[%i, %j] : !zap.tensor<[?, ?], f32>
    }
  }

  kgen.return
}
