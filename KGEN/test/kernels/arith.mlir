// RUN: kgen-opt %s -elaborate-kernels="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

kgen.generator.interface @buffer.loadOrValue<isLoad: i1, type: dtype>(%buffer: !meta.buffer<?, type>, %idx: index, %val: !meta.scalar<type>) -> !meta.scalar<type>

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

// Add two buffers. Parameter 'bcst' indicates if the first buffer should be broadcasted.
kgen.generator.interface @add_bcst<bcst: i1, type: dtype>(%in1: !meta.buffer<?, type>, %in2: !meta.buffer<?, type>,
  %out : !meta.buffer<?, type>)

kgen.generator @add_scalar_loop<bcst: i1, type: dtype>(%in1: !meta.buffer<?, type>, %in2: !meta.buffer<?, type>,
  %out : !meta.buffer<?, type>)
  implements @add_bcst {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index

  // TODO: Must assert that buffers have the same size or we are doing broadcast.
  %size = meta.buffer.size %out: !meta.buffer<?, type>

  // Using 0 as a placeholder for undefined value since we do not have optional values.
  // %undef will be eliminated after kernel elaboration and simplification.
  %undef = pop.constant(0) : !meta.scalar<type>
  kgen.param.declare no_bcst: i1 = <not(bcst)>
  %bcst_val =  kgen.call @buffer.loadOrValue<isLoad:i1=bcst, type:dtype=type>(%in1, %zero, %undef) : (!meta.buffer<?, type>, index, !meta.scalar<type>) -> !meta.scalar<type>

  scf.for %i = %zero to %size step %one {
      %src1 = kgen.call @buffer.loadOrValue<isLoad:i1=no_bcst, type:dtype=type>(%in1, %i, %bcst_val) : (!meta.buffer<?, type>, index, !meta.scalar<type>) -> !meta.scalar<type>
      %src2 = pop.buffer.load %in2[%i] : !meta.buffer<?, type>
      %res = pop.add %src1, %src2 : !meta.scalar<type>
      pop.buffer.store %res, %out[%i] : !meta.buffer<?, type>
  }
  kgen.return
}

// Add two buffers applying implicit broadcast rules.
kgen.generator.interface @add<type: dtype>(%in1: !meta.buffer<?, type>, %in2: !meta.buffer<?, type>, %out : !meta.buffer<?, type>)

kgen.generator @add_impl<type: dtype>(%in1:  !meta.buffer<?, type>, %in2: !meta.buffer<?, type>, %out : !meta.buffer<?, type>)
  implements @add {
  // If %in1.size == 1
  %one = arith.constant 1 : index
  %size1 = meta.buffer.size %in1 : !meta.buffer<?, type>
  %broadcastLeft = arith.cmpi eq, %size1, %one : index
  scf.if %broadcastLeft {
    // Broadcast first input buffer
    kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in1, %in2, %out) : (!meta.buffer<?, type>, !meta.buffer<?, type>, !meta.buffer<?, type>) -> ()
  } else {
    // If %in2.size == 1
    %size2 = meta.buffer.size %in2 : !meta.buffer<?, type>
    %broadcastRight = arith.cmpi eq, %size2, %one : index
    scf.if %broadcastRight {
      // Broadcast second input buffer. Addition is commutative, so just swap operands.
      kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in2, %in1, %out) : (!meta.buffer<?, type>, !meta.buffer<?, type>, !meta.buffer<?, type>) -> ()
    } else {
      // No broadcast case
      kgen.call @add_bcst<bcst:i1=0, type:dtype=type>(%in1, %in2, %out) : (!meta.buffer<?, type>, !meta.buffer<?, type>, !meta.buffer<?, type>) -> ()
    }
  }
  kgen.return
}

// Instantiate @add for f32

// CHECK-LABEL: kgen.kernel @add_f32_kernel(%arg0: !meta.buffer<?, f32>, %arg1: !meta.buffer<?, f32>, %arg2: !meta.buffer<?, f32>)
// CHECK: kgen.call @"add_impl,type=f32"(%arg0, %arg1, %arg2) : (!meta.buffer<?, f32>, !meta.buffer<?, f32>, !meta.buffer<?, f32>) -> ()
kgen.generator @add_f32(%in1: !meta.buffer<?, f32>, %in2: !meta.buffer<?, f32>, %out: !meta.buffer<?, f32>) {
  kgen.call @add<type: dtype = f32>(%in1, %in2, %out) : (!meta.buffer<?, f32>, !meta.buffer<?, f32>, !meta.buffer<?, f32>) -> ()
  kgen.return
}
