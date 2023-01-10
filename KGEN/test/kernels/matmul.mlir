// RUN: kgen-opt %s -lower-lit -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"


kgen.generator.interface @index2D(%row: index, %col: index, %stride: index) -> index


kgen.generator.interface @matmul<type: dtype>(
    %A: !pop.pointer<scalar<type>>,
    %B: !pop.pointer<scalar<type>>,
    %C: !pop.pointer<scalar<type>>,
    %M: index,
    %N: index,
    %K: index)

// Implements a naive matrix multiplication kernel.
lit.func @matmaul_naive<type: dtype>(
    %A: !pop.pointer<scalar<type>>,
    %B: !pop.pointer<scalar<type>>,
    %C: !pop.pointer<scalar<type>>,
    %M: index,
    %N: index,
    %K: index)
    implements @matmul {
  %zero = index.constant 0
  %one = index.constant 1
  scf.for %i = %zero to %M step %one {
    scf.for %j = %zero to %N step %one {
      %zero_si64 = kgen.param.constant: !pop.scalar<si64> = <#pop.simd<0>>
      %init = pop.cast %zero_si64 : !pop.scalar<si64> to !pop.scalar<type>
      %acc = scf.for %k = %zero to %K step %one iter_args(%sum = %init) -> (!pop.simd<1, type>) {
        %aikIndex = kgen.call @index2D(%i, %k, %N) : (index, index, index) -> index
        %bkjIndex = kgen.call @index2D(%k, %j, %M) : (index, index, index) -> index
        %aPtr = pop.offset %A[%aikIndex] : !pop.pointer<scalar<type>>
        %bPtr = pop.offset %B[%bkjIndex] : !pop.pointer<scalar<type>>
        %aik = pop.load %aPtr : !pop.pointer<scalar<type>>
        %bkj = pop.load %bPtr : !pop.pointer<scalar<type>>
        %res = pop.fma %aik, %bkj, %sum : !pop.simd<1, type>
        scf.yield %res : !pop.simd<1, type>
      }
      %cij = kgen.call @index2D(%i, %j, %K) : (index, index, index) -> index
      %cPtr = pop.offset %C[%cij] : !pop.pointer<scalar<type>>
      pop.store %acc, %cPtr : !pop.pointer<scalar<type>>
    }
  }
  kgen.return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.func @matmul_f32
// CHECK: kgen.call @"matmaul_naive,type=f32"
kgen.generator @matmul_f32(
    %A: !pop.pointer<scalar<f32>>,
    %B: !pop.pointer<scalar<f32>>,
    %C: !pop.pointer<scalar<f32>>,
    %M: index,
    %N: index,
    %K: index) {
  kgen.call @matmul<type: dtype = f32>(%A, %B, %C, %M, %N, %K) :
    (!pop.pointer<scalar<f32>>, !pop.pointer<scalar<f32>>, !pop.pointer<scalar<f32>>,
     index, index, index) -> ()
  kgen.return
}
