// RUN: kgen-opt -elaborate-generators %s -verify-diagnostics -split-input-file

// expected-note @below {{failed to interpret function @out_of_range_read}}
kgen.generator @out_of_range_read() -> i32 {
  %0 = pop.stack_allocation 0 x i32
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{memory access size 4 is out-of-bounds}}
  %1 = pop.load %0 : !kgen.pointer<i32>
  kgen.return %1 : i32
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  %0 = kgen.param.constant: i32 = <apply(:() -> i32 @out_of_range_read)>
  kgen.return
}

// -----

kgen.generator @return_stack_addr() -> !kgen.pointer<index> {
  %0 = pop.stack_allocation 1 x index
  kgen.return %0 : !kgen.pointer<index>
}

// expected-note @below {{failed to interpret function @stack_use_after_free}}
kgen.generator @stack_use_after_free() -> index {
  %0 = kgen.call @return_stack_addr() : () -> !kgen.pointer<index>
  // expected-note @below {{failed to interpret operation pop.load(#interp.pointer}}
  // expected-note @below {{address is out-of-bounds}}
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant = <apply(:() -> index @stack_use_after_free)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @heap_use_after_free}}
kgen.generator @heap_use_after_free() -> i64 {
  %idx32 = index.constant 32
  %idx8 = index.constant 8
  %0 = pop.aligned_alloc %idx32, %idx8 : <i64>
  pop.aligned_free %0 : <i64>
  // expected-note @below {{failed to interpret operation pop.load(#interp.pointer}}
  // expected-note @below {{accessing memory that was freed}}
  %1 = pop.load %0 : !kgen.pointer<i64>
  kgen.return %1 : i64
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant: i64 = <apply(:() -> i64 @heap_use_after_free)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @clobber_pointer}}
kgen.generator @clobber_pointer(%arg0: i16) -> i16 {
  %0 = pop.stack_allocation 1 x i64
  %1 = pop.stack_allocation 1 x !kgen.pointer<i64>
  pop.store %0, %1 : !kgen.pointer<pointer<i64>>
  %2 = pop.pointer.bitcast %1 : !kgen.pointer<pointer<i64>> to !kgen.pointer<i16>
  %idx1 = index.constant 1
  %3 = pop.offset %2[%idx1] : !kgen.pointer<i16>
  // expected-note @below {{failed to interpret operation pop.store}}
  // expected-note @below {{write clobbers a pointer region}}
  pop.store %arg0, %3 : !kgen.pointer<i16>
  kgen.return %arg0 : i16
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant: i16 = <apply(:(i16) -> i16 @clobber_pointer, 5)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @int_literal_convert}}
kgen.generator @int_literal_convert() -> si64 {

  %0 = kgen.param.constant: !kgen.int_literal = <36893488147419103232>
  // expected-note @below {{failed to interpret operation kgen.int_literal.convert(#kgen.int_literal<36893488147419103232> : !kgen.int_literal)}}
  // expected-note @below {{integer value 36893488147419103232 requires 67 bits to store, but the destination bit width is only 64 bits wide}}
  %1 = kgen.int_literal.convert %0 : to si64
  kgen.return %1 : si64
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_convert() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant: si64 = <apply(:() -> si64 @int_literal_convert)>
  kgen.return
}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @clobber_pointer}}
kgen.generator @clobber_pointer(%arg0: i64) -> i64 {
  %0 = pop.stack_allocation 1 x i64
  %1 = pop.stack_allocation 1 x !kgen.struct<(i64, !kgen.pointer<i64>)>
  %2 = kgen.struct.gep %1[1] : <struct<(i64, !kgen.pointer<i64>)>>
  pop.store %0, %2 : !kgen.pointer<pointer<i64>>
  %3 = kgen.struct.gep %1[0] : <struct<(i64, !kgen.pointer<i64>)>>
  %4 = pop.pointer_to_index %3 : !kgen.pointer<i64> to !pop.scalar<index>
  %idx2 = kgen.param.constant: scalar<index> = <1>
  %5 = pop.add %4, %idx2 : !pop.scalar<index>
  %6 = pop.index_to_pointer %5 : !pop.scalar<index> to !kgen.pointer<i64>
  // expected-note @below {{failed to interpret operation pop.store}}
  // expected-note @below {{write clobbers a pointer region}}
  pop.store %arg0, %6 : !kgen.pointer<i64>
  kgen.return %arg0 : i64
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant: i64 = <apply(:(i64) -> i64 @clobber_pointer, 1)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @parameter_closure}}
kgen.generator @parameter_closure() -> index {
  // expected-note @below {{failed to interpret operation pop.compiler.global_load{name: "named_global"}()}}
  // expected-note @below {{cannot evaluate standalone capturing closure at compile time}}
  %0 = pop.compiler.global_load "named_global" : index
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @use_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant = <apply(:() -> index @parameter_closure)>
  kgen.return
}

}
