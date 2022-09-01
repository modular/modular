// RUN: kgen-opt %s -lower-hlkgen -elaborate-kernels="search-path=%S" | FileCheck %s

kgen.include "library.mlir"


kgen.generator.interface @index2D(%row: index, %col: index, %stride: index) -> index


kgen.generator.interface @matmul<type: dtype>(
    %A: !meta.buffer<?, type>,
    %B: !meta.buffer<?, type>,
    %C: !meta.buffer<?, type>,
    %M: index,
    %N: index,
    %K: index)

// Implements a naive matrix multiplication kernel.
hlkgen.generator @matmaul_naive<type: dtype>(
    %A: !meta.buffer<?, type>,
    %B: !meta.buffer<?, type>,
    %C: !meta.buffer<?, type>,
    %M: index,
    %N: index,
    %K: index)
    implements @matmul {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %init = pop.constant(0) : !meta.scalar<type>
      %acc = scf.for %k = %zero to %K step %one iter_args(%sum = %init) -> (!meta.scalar<type>) {
        %aikIndex = kgen.call @index2D(%i, %k, %N) : (index, index, index) -> index
        %bkjIndex = kgen.call @index2D(%k, %j, %M) : (index, index, index) -> index
        %aik = pop.buffer.load %A[%aikIndex] : !meta.buffer<?, type>
        %bkj = pop.buffer.load %B[%bkjIndex] : !meta.buffer<?, type>
        %res = pop.fma %aik, %bkj, %sum : !meta.scalar<type>
        scf.yield %res : !meta.scalar<type>
      }
      %cij = kgen.call @index2D(%i, %j, %K) : (index, index, index) -> index
      pop.buffer.store %acc, %C[%cij] : !meta.buffer<?, type>
    }
  }
  kgen.return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.kernel @matmul_f32
// CHECK: kgen.call @"matmaul_naive,type=f32"
kgen.generator @matmul_f32(
    %A: !meta.buffer<?, f32>,
    %B: !meta.buffer<?, f32>,
    %C: !meta.buffer<?, f32>,
    %M: index,
    %N: index,
    %K: index) {
  kgen.call @matmul<type: dtype = f32>(%A, %B, %C, %M, %N, %K) :
    (!meta.buffer<?, f32>, !meta.buffer<?, f32>, !meta.buffer<?, f32>, index, index, index) -> ()
  kgen.return
}
