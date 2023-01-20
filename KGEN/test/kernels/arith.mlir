// RUN: kgen-opt %s -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

kgen.generator.interface @buffer.loadOrValue<isLoad: i1, type: dtype>(
  %ptr: !pop.pointer<scalar<type>>, %idx: index, %val: !pop.scalar<type>) -> !pop.scalar<type>

//===----------------------------------------------------------------------===//
// add
//===----------------------------------------------------------------------===//

// Add two buffers. Parameter 'bcst' indicates if the first buffer should be broadcasted.
kgen.generator.interface @add_bcst<bcst: i1, type: dtype>(
    %in1: !pop.pointer<scalar<type>>, %in2: !pop.pointer<scalar<type>>,
    %out: !pop.pointer<scalar<type>>, %size: index)

kgen.generator @add_scalar_loop<bcst: i1, type: dtype>(
    %in1: !pop.pointer<scalar<type>>, %in2: !pop.pointer<scalar<type>>,
    %out: !pop.pointer<scalar<type>>, %size: index)
  implements @add_bcst {
  %zero = index.constant 0
  %one = index.constant 1

  // Using 0 as a placeholder for undefined value since we do not have optional values.
  // %undef will be eliminated after kernel elaboration and simplification.
  %zero_si64 = kgen.param.constant: !pop.scalar<si64> = <<0>>
  %undef = pop.cast %zero_si64 : !pop.scalar<si64> to !pop.scalar<type>
  kgen.param.declare no_bcst: i1 = <not(bcst)>
  %bcst_val =  kgen.call @buffer.loadOrValue<isLoad:i1=bcst, type:dtype=type>(%in1, %zero, %undef)
    : (!pop.pointer<scalar<type>>, index, !pop.scalar<type>) -> !pop.scalar<type>

  scf.for %i = %zero to %size step %one {
      %src1 = kgen.call @buffer.loadOrValue<isLoad:i1=no_bcst, type:dtype=type>(%in1, %i, %bcst_val)
        : (!pop.pointer<scalar<type>>, index, !pop.scalar<type>) -> !pop.scalar<type>
      %inPtr = pop.offset %in2[%i] : !pop.pointer<scalar<type>>
      %src2 = pop.load %inPtr : !pop.pointer<scalar<type>>
      %res = pop.add %src1, %src2 : !pop.scalar<type>
      %outPtr = pop.offset %out[%i] : !pop.pointer<scalar<type>>
      pop.store %res, %outPtr : !pop.pointer<scalar<type>>
  }
  kgen.return
}

// Add two buffers applying implicit broadcast rules.
kgen.generator.interface @add<type: dtype>(
    %in1: !pop.pointer<scalar<type>>, %in2: !pop.pointer<scalar<type>>,
    %size1: index, %size2: index, %out: !pop.pointer<scalar<type>>)

kgen.generator @add_impl<type: dtype>(
    %in1: !pop.pointer<scalar<type>>, %in2: !pop.pointer<scalar<type>>,
    %size1: index, %size2: index, %out: !pop.pointer<scalar<type>>)
  implements @add {
  // If %in1.size == 1
  %one = index.constant 1
  %broadcastLeft = index.cmp eq(%size1, %one)
  scf.if %broadcastLeft {
    // Broadcast first input buffer
    kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in1, %in2, %out, %size1)
      : (!pop.pointer<scalar<type>>, !pop.pointer<scalar<type>>,
         !pop.pointer<scalar<type>>, index) -> ()
  } else {
    // If %in2.size == 1
    %broadcastRight = index.cmp eq(%size2, %one)
    scf.if %broadcastRight {
      // Broadcast second input buffer. Addition is commutative, so just swap operands.
      kgen.call @add_bcst<bcst:i1=1, type:dtype=type>(%in2, %in1, %out, %size2)
      : (!pop.pointer<scalar<type>>, !pop.pointer<scalar<type>>,
         !pop.pointer<scalar<type>>, index) -> ()
    } else {
      // No broadcast case
      kgen.call @add_bcst<bcst:i1=0, type:dtype=type>(%in1, %in2, %out, %size1)
      : (!pop.pointer<scalar<type>>, !pop.pointer<scalar<type>>,
         !pop.pointer<scalar<type>>, index) -> ()
    }
  }
  kgen.return
}

// Instantiate @add for f32

// CHECK-LABEL: kgen.func @add_f32
// CHECK: kgen.call @"add_impl,type=f32"(%arg0, %arg1, %arg2, %arg3, %arg4)
// CHECK-SAME: (!pop.pointer<scalar<f32>>, !pop.pointer<scalar<f32>>, index, index, !pop.pointer<scalar<f32>>) -> ()
kgen.generator @add_f32(
    %in1: !pop.pointer<scalar<f32>>, %in2: !pop.pointer<scalar<f32>>,
    %size1: index, %size2: index, %out: !pop.pointer<scalar<f32>>) {
  kgen.call @add<type: dtype = f32>(%in1, %in2, %size1, %size2, %out)
    : (!pop.pointer<scalar<f32>>, !pop.pointer<scalar<f32>>,
       index, index, !pop.pointer<scalar<f32>>) -> ()
  kgen.return
}
