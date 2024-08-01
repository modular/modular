// RUN: kgen-opt -elaborate-generators %s -verify-diagnostics -split-input-file

// expected-note @below {{failed to interpret function @out_of_range_read}}
kgen.generator @out_of_range_read() -> i32 {
  %0 = pop.stack_allocation 0 x i32
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{memory access size 4 is out-of-bounds}}
  %1 = pop.load %0 : !kgen.pointer<i32>
  kgen.return %1 : i32
}

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_convert() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: si64 = <apply(:() -> si64 @int_literal_convert)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @int_literal_convert_unsigned}}
kgen.generator @int_literal_convert_unsigned() -> ui64 {

  %0 = kgen.param.constant: !kgen.int_literal = <-1>
  // expected-note @below {{failed to interpret operation kgen.int_literal.convert(#kgen.int_literal<-1> : !kgen.int_literal)}}
  // expected-note @below {{integer value -1 is negative, but is being converted to an unsigned type.}}
  %1 = kgen.int_literal.convert %0 : to ui64
  kgen.return %1 : ui64
}

kgen.generator @call_convert_unsigned() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: ui64 = <apply(:() -> ui64 @int_literal_convert_unsigned)>
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
  %4 = pop.pointer.bitcast %3 : !kgen.pointer<i64> to !kgen.pointer<i32>
  %idx1 = index.constant 1
  %5 = pop.offset %4[%idx1] : !kgen.pointer<i32>
  %6 = pop.pointer.bitcast %5 : !kgen.pointer<i32> to !kgen.pointer<i64>
  // expected-note @below {{failed to interpret operation pop.store}}
  // expected-note @below {{write clobbers a pointer region}}
  pop.store %arg0, %6 : !kgen.pointer<i64>
  kgen.return %arg0 : i64
}

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator export @use_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @parameter_closure)>
  kgen.return
}

}

// -----

module attributes {M.target = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// expected-note @below {{failed to interpret function @load_union}}
kgen.generator @load_union() -> !pop.union<index> {
  %0 = pop.stack_allocation 1 x union<index>
  %1 = kgen.param.constant: union<index> = <{42}>
  pop.store %1, %0 : !kgen.pointer<union<index>>
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{cannot read a union-typed value}}
  %2 = pop.load %0 : !kgen.pointer<union<index>>
  kgen.return %2 : !pop.union<index>
}

kgen.generator export @use_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.constant: union<index> = <apply(:() -> !pop.union<index> @load_union)>
  kgen.return
}

}
