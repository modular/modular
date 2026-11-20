// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -verify-diagnostics -split-input-file
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-PARAM

// COM: use-parametric-interpret=true has slight difference from =false for error messages.
//      Using FileCheck instead to check those with CHECK-PRAMA prefix.

 module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-PARAM: failed to interpret function @minus
// expected-note @below {{failed to interpret function @minus}}
kgen.generator @minus(%arg0: index, %arg1: index) -> index {
  // CHECK-PARAM: failed to interpret operation index.sub(4294967295 : index, -4294967295 : index)
  // CHECK-PARAM: value '4294967295' of the operation `index.sub` is too large for 32-bit index
  // expected-note @below {{failed to interpret operation index.sub(4294967295 : index, -4294967295 : index)}}
  // expected-note @below {{value '4294967295' of the operation `index.sub` is too large for 32-bit index}}
  %0 = index.sub %arg0, %arg1
  hlcf.return %0 : index
}

// CHECK-LABEL: kgen.func @callIt
// expected-error @+1 {{function instantiation failed}}
kgen.generator export @callIt() -> index {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @minus, 0x00000000ffffffff, -4294967295)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}
}

// -----

// COM: Shifting too many bits for a 32-bit target - undefined behaviour

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @shr_index}}
kgen.generator @shr_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
// expected-note @below {{failed to interpret operation pop.shr(#kgen<simd 3> : !kgen.scalar<index>, #kgen<simd 63> : !kgen.scalar<index>)}}
// expected-note @below {{failed to interpret pop.shr}}
  %0 = pop.shr %arg0, %arg1 : !kgen.scalar<index>
  hlcf.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShr
// expected-error @below {{function instantiation failed}}
kgen.generator export @testShr() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <3>
  kgen.param.declare S1: scalar<index> = <63>
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shr_index, S0, S1)>
  // CHECK: = <0>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  hlcf.return %0 : !kgen.scalar<index>
}

}

// -----

// COM: Shifting too many bits for a 64-bit target - undefined behaviour

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @shl_index}}
kgen.generator @shl_index(%arg0 : !kgen.scalar<index>, %arg1 : !kgen.scalar<index>) -> !kgen.scalar<index> {
// expected-note @below {{failed to interpret operation pop.shl(#kgen<simd 3> : !kgen.scalar<index>, #kgen<simd 65> : !kgen.scalar<index>)}}
// expected-note @below {{failed to interpret pop.shl}}
  %0 = pop.shl %arg0, %arg1 : !kgen.scalar<index>
  hlcf.return %0 : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func export @testShl
// expected-error @below {{function instantiation failed}}
kgen.generator export @testShl() -> !kgen.scalar<index> {
  kgen.param.declare S0: scalar<index> = <3>
  kgen.param.declare S1: scalar<index> = <65>
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare S2: scalar<index> = <apply(:(!kgen.scalar<index>, !kgen.scalar<index>) -> !kgen.scalar<index> @shl_index, S0, S1)>
  // CHECK: = <0>
  %0 = kgen.param.constant: !kgen.scalar<index> = <S2>
  hlcf.return %0 : !kgen.scalar<index>
}

}

// -----

// COM: Division by zero must be diagnosed rather than trapping in APInt.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @rems_index}}
kgen.generator @rems_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.rems(5 : index, 0 : index)}}
  // expected-note @below {{`index.rems` failed due to division by zero}}
  %0 = index.rems %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @testRemsByZero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @rems_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @remu_index}}
kgen.generator @remu_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.remu(5 : index, 0 : index)}}
  // expected-note @below {{`index.remu` failed due to division by zero}}
  %0 = index.remu %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @test_remu_by_zero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @remu_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @divs_index}}
kgen.generator @divs_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.divs(5 : index, 0 : index)}}
  // expected-note @below {{`index.divs` failed due to division by zero}}
  %0 = index.divs %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @test_divs_by_zero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @divs_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @divu_index}}
kgen.generator @divu_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.divu(5 : index, 0 : index)}}
  // expected-note @below {{`index.divu` failed due to division by zero}}
  %0 = index.divu %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @test_divu_by_zero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @divu_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @ceildivs_index}}
kgen.generator @ceildivs_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.ceildivs(5 : index, 0 : index)}}
  // expected-note @below {{`index.ceildivs` failed due to division by zero}}
  %0 = index.ceildivs %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @test_ceildivs_by_zero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @ceildivs_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}

// -----

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {

// expected-note @below {{failed to interpret function @ceildivu_index}}
kgen.generator @ceildivu_index(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.ceildivu(5 : index, 0 : index)}}
  // expected-note @below {{`index.ceildivu` failed due to division by zero}}
  %0 = index.ceildivu %arg0, %arg1
  hlcf.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @test_ceildivu_by_zero() -> index {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @ceildivu_index, 5, 0)>
  %0 = kgen.param.constant: index = <value>
  hlcf.return %0 : index
}

}
