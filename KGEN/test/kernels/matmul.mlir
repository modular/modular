// RUN: kgen-opt %s -lower-lit -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"


kgen.generator.interface @index2D(%row: index, %col: index, %stride: index) -> index


kgen.generator.interface @matmul<type: dtype>(
    %A: !zap.buffer<?, type>,
    %B: !zap.buffer<?, type>,
    %C: !zap.buffer<?, type>,
    %M: index,
    %N: index,
    %K: index)

// Implements a naive matrix multiplication kernel.
lit.func @matmaul_naive<type: dtype>(
    %A: !zap.buffer<?, type>,
    %B: !zap.buffer<?, type>,
    %C: !zap.buffer<?, type>,
    %M: index,
    %N: index,
    %K: index)
    implements @matmul {
  %zero = index.constant 0
  %one = index.constant 1
  %undef = pop.constant(0) : !pop.scalar<type>
  %undefVec = pop.simd.splat %undef : !pop.simd<1, type>
  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %init = pop.constant(0) : !pop.scalar<type>
      %acc = scf.for %k = %zero to %K step %one iter_args(%sum = %init) -> (!pop.scalar<type>) {
        %aikIndex = kgen.call @index2D(%i, %k, %N) : (index, index, index) -> index
        %bkjIndex = kgen.call @index2D(%k, %j, %M) : (index, index, index) -> index
        %aik0 = zap.buffer.load %A[%aikIndex] : !zap.buffer<?, type>, !pop.simd<1, type>
        %aik = pop.simd.extractelement %aik0[%zero] : !pop.simd<1, type>
        %bkj0 = zap.buffer.load %B[%bkjIndex] : !zap.buffer<?, type>, !pop.simd<1, type>
        %bkj = pop.simd.extractelement %bkj0[%zero] : !pop.simd<1, type>
        %res = pop.fma %aik, %bkj, %sum : !pop.scalar<type>
        scf.yield %res : !pop.scalar<type>
      }
      %cij = kgen.call @index2D(%i, %j, %K) : (index, index, index) -> index
      %accVec = pop.simd.insertelement %acc, %undefVec[%zero] : !pop.simd<1, type>
      zap.buffer.store %accVec, %C[%cij] : !pop.simd<1, type>, !zap.buffer<?, type>
    }
  }
  kgen.return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.func @matmul_f32
// CHECK: kgen.call @"matmaul_naive,type=f32"
kgen.generator @matmul_f32(
    %A: !zap.buffer<?, f32>,
    %B: !zap.buffer<?, f32>,
    %C: !zap.buffer<?, f32>,
    %M: index,
    %N: index,
    %K: index) {
  kgen.call @matmul<type: dtype = f32>(%A, %B, %C, %M, %N, %K) :
    (!zap.buffer<?, f32>, !zap.buffer<?, f32>, !zap.buffer<?, f32>, index, index, index) -> ()
  kgen.return
}
