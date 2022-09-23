// RUN: kgen-execute %s -execute -func="for_loop:f32():%t.o" | FileCheck %s --check-prefix=FOR
// RUN: kgen-execute %s -execute -func="while_loop:f32():%t.o" | FileCheck %s --check-prefix=WHILE
// RUN: kgen-execute %s -execute -func="while_accum_loop:f32():%t.o" | FileCheck %s --check-prefix=WHILE_ACCUM

kgen.func public @for_loop() -> f32 {
  %av = pop.constant(1.0 : f32) : !pop.scalar<f32>
  %c10 = pop.constant(10.0 : f32) : !pop.scalar<f32>
  %lb = index.constant 0
  %ub = index.constant 10
  %step = index.constant 1
  %rv = scf.for %i = %lb to %ub step %step iter_args(%v = %av) -> (!pop.scalar<f32>) {
    %n = pop.add %c10, %v : !pop.scalar<f32>
    scf.yield %n : !pop.scalar<f32>
  }
  %r = pop.type_lower %rv : !pop.scalar<f32> to f32
  kgen.return %r : f32
}

kgen.func public @while_loop() -> f32 {
  %init = pop.constant(1.2 : f32) : !pop.scalar<f32>
  %limit = pop.constant(10. : f32) : !pop.scalar<f32>
  %result = scf.while (%v = %init) : (!pop.scalar<f32>) -> !pop.scalar<f32> {
    %cmp = pop.cmp lt(%v, %limit) : !pop.scalar<f32>
    %cond = pop.type_lower %cmp : !pop.scalar<bool> to i1
    scf.condition(%cond) %v : !pop.scalar<f32>
  } do {
  ^bb0(%u : !pop.scalar<f32>):
    %next = pop.mul %u, %u : !pop.scalar<f32>
    scf.yield %next : !pop.scalar<f32>
  }
  %res = pop.type_lower %result : !pop.scalar<f32> to f32
  kgen.return %res : f32
}

// Performs the following operations 0+1+...+12 (should return 78)
// int size = 13;
// int iter = 0;
// float accum = 0;
// while (iter <= size-8) {
//     int ii = iter;
//     while (ii < iter + 8) {
//         accum += ii;
//         ii++;
//     }
//     iter += 8;
// }
// while (iter < size) {
//     accum += iter;
//     iter++;
// }
// return accum;
kgen.func public @while_accum_loop() -> f32 {
  %size = index.constant 13
  %one = index.constant 1
  %eight = index.constant 8
  %iter_init = index.constant 0
  %size_minus_8 = index.sub %size, %eight
  %accum_init = pop.constant(0.0 :f32) : !pop.scalar<f32>
  // while (iter+8 < size) {
  //     int ii = iter;
  //     while (ii < iter + 8) {
  //         accum += ii;
  //         ii++;
  //     }
  //     iter += 8;
  // }
  %new_index, %accum0 = scf.while (%iter = %iter_init, %accum = %accum_init) : (index, !pop.scalar<f32>) -> (index, !pop.scalar<f32>) {
    %cond = index.cmp sle(%iter, %size_minus_8)
    scf.condition(%cond) %iter, %accum : index, !pop.scalar<f32>
  } do {
  ^bb0(%iter1: index, %accum1: !pop.scalar<f32>):
    %ii_last = index.add %iter1, %eight
    //     while (ii < iter + 8) {
    //         accum += ii;
    //         ii++;
    //     }
    %new_index, %accum2 = scf.while (%ii = %iter1, %accum = %accum1) : (index, !pop.scalar<f32>) -> (index, !pop.scalar<f32>) {
      %cond = index.cmp slt(%ii, %ii_last)
      scf.condition(%cond) %ii, %accum : index, !pop.scalar<f32>
    } do {
    ^bb1(%ii: index, %accum2: !pop.scalar<f32>):
      %iiInt32 = index.casts %ii : index to i32
      %iiMetaInt32 = pop.type_raise %iiInt32 : i32 to !pop.scalar<si32>
      %iiFloat32 = pop.cast %iiMetaInt32 : !pop.scalar<si32> to !pop.scalar<f32>
      %accum3 = pop.add %accum2, %iiFloat32 : !pop.scalar<f32>
      %next_index = index.add %ii, %one
      scf.yield %next_index, %accum3 : index, !pop.scalar<f32>
    }
    scf.yield %ii_last, %accum2 : index, !pop.scalar<f32>
  }
  // while (iter < size) {
  //     accum += iter;
  //     iter++;
  // }
  %iter1, %accum = scf.while (%iter = %new_index, %accum = %accum0) : (index, !pop.scalar<f32>) -> (index, !pop.scalar<f32>) {
    %cond = index.cmp slt(%iter, %size)
    scf.condition(%cond) %iter, %accum : index, !pop.scalar<f32>
  } do {
  ^bb1(%ii: index, %accum2: !pop.scalar<f32>):
    %iiInt32 = index.casts %ii : index to i32
    %iiMetaInt32 = pop.type_raise %iiInt32 : i32 to !pop.scalar<si32>
    %iiFloat32 = pop.cast %iiMetaInt32 : !pop.scalar<si32> to !pop.scalar<f32>
    %accum3 = pop.add %accum2, %iiFloat32 : !pop.scalar<f32>
    %next_index = index.add %ii, %one
    scf.yield %next_index, %accum3 : index, !pop.scalar<f32>
  }
  %res = pop.type_lower %accum : !pop.scalar<f32> to f32
  kgen.return %res : f32
}



// FOR: --- 'for_loop' returned 101.{{[0-9]+}}
// WHILE: --- 'while_loop' returned 18.4{{[0-9]+}}
// WHILE_ACCUM: --- 'while_accum_loop' returned 78.{{[0]+}}
