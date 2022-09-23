// RUN: kgen-opt %s -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

kgen.generator.interface @buffer.loadOrValue<isLoad: i1, type: dtype>(%buffer: !zap.buffer<?, type>, %idx: index, %val: !meta.scalar<type>) -> !meta.scalar<type>

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

// Add two buffers. Parameter 'bcst' indicates if the first buffer should be broadcasted.
kgen.generator.interface @add_bcst<bcst: i1, type: dtype>(%in1: !zap.buffer<?, type>, %in2: !zap.buffer<?, type>,
  %out : !zap.buffer<?, type>)

kgen.generator @add_scalar_loop<bcst: i1, type: dtype>(%in1: !zap.buffer<?, type>, %in2: !zap.buffer<?, type>,
  %out : !zap.buffer<?, type>)
  implements @add_bcst {
  %zero = index.constant 0
  %one = index.constant 1

  // TODO: Must assert that buffers have the same size or we are doing broadcast.
  %size = zap.buffer.size %out: !zap.buffer<?, type>

  // Using 0 as a placeholder for undefined value since we do not have optional values.
  // %undef will be eliminated after kernel elaboration and simplification.
  %undef = pop.constant(0) : !meta.scalar<type>
  kgen.param.declare no_bcst: i1 = <not(bcst)>
  %bcst_val =  kgen.call @buffer.loadOrValue<isLoad:i1=bcst, type:dtype=type>(%in1, %zero, %undef) : (!zap.buffer<?, type>, index, !meta.scalar<type>) -> !meta.scalar<type>

  scf.for %i = %zero to %size step %one {
      %src1 = kgen.call @buffer.loadOrValue<isLoad:i1=no_bcst, type:dtype=type>(%in1, %i, %bcst_val) : (!zap.buffer<?, type>, index, !meta.scalar<type>) -> !meta.scalar<type>
      %src2 = zap.buffer.load %in2[%i] : !zap.buffer<?, type>
      %res = pop.add %src1, %src2 : !meta.scalar<type>
      zap.buffer.store %res, %out[%i] : !zap.buffer<?, type>
  }
  kgen.return
}

// Add two buffers applying implicit broadcast rules.
kgen.generator.interface @add<type: dtype>(%in1: !zap.buffer<?, type>, %in2: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)

kgen.generator @add_impl<type: dtype>(%in1:  !zap.buffer<?, type>, %in2: !zap.buffer<?, type>, %out : !zap.buffer<?, type>)
  implements @add {
  // If %in1.size == 1
  %one = index.constant 1
  %size1 = zap.buffer.size %in1 : !zap.buffer<?, type>
  %broadcastLeft = index.cmp eq(%size1, %one)
  scf.if %broadcastLeft {
    // Broadcast first input buffer
    kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in1, %in2, %out) : (!zap.buffer<?, type>, !zap.buffer<?, type>, !zap.buffer<?, type>) -> ()
  } else {
    // If %in2.size == 1
    %size2 = zap.buffer.size %in2 : !zap.buffer<?, type>
    %broadcastRight = index.cmp eq(%size2, %one)
    scf.if %broadcastRight {
      // Broadcast second input buffer. Addition is commutative, so just swap operands.
      kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in2, %in1, %out) : (!zap.buffer<?, type>, !zap.buffer<?, type>, !zap.buffer<?, type>) -> ()
    } else {
      // No broadcast case
      kgen.call @add_bcst<bcst:i1=0, type:dtype=type>(%in1, %in2, %out) : (!zap.buffer<?, type>, !zap.buffer<?, type>, !zap.buffer<?, type>) -> ()
    }
  }
  kgen.return
}

// Instantiate @add for f32

// CHECK-LABEL: kgen.func @add_f32
// CHECK-SAME: %[[ARG0:.*]]: !zap.buffer<?, f32>, %[[ARG1:.*]]: !zap.buffer<?, f32>, %[[ARG2:.*]]: !zap.buffer<?, f32>
// CHECK: kgen.call @"add_impl,type=f32"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!zap.buffer<?, f32>, !zap.buffer<?, f32>, !zap.buffer<?, f32>) -> ()
kgen.generator @add_f32(%in1: !zap.buffer<?, f32>, %in2: !zap.buffer<?, f32>, %out: !zap.buffer<?, f32>) {
  kgen.call @add<type: dtype = f32>(%in1, %in2, %out) : (!zap.buffer<?, f32>, !zap.buffer<?, f32>, !zap.buffer<?, f32>) -> ()
  kgen.return
}
