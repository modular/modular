// RUN: kgen-opt %s -elaborate-generators='sanitize-symbols=true' | FileCheck %s

// CHECK: [[F0:@_mojo_nutty_names_int_Int_int_Int_v_1_[a-zA-Z0-9_]+]]()
// CHECK: [[F1:@_mojo_nutty_names_int_Int_int_Int_v_8_[a-zA-Z0-9_]+]]()
kgen.generator @"$mojo::nutty::names[$int::Int]($int::\22Int\22)"<v>() {
  kgen.return
}

// CHECK: [[F2:@_no_concrete_plz_[a-zA-Z0-9_]+]]()
kgen.generator @"$no::concrete::plz()"() {
  kgen.return
}

// CHECK: kgen.func export @a_VALID_name123()
kgen.generator export @a_VALID_name123() {
  // CHECK: call [[F0]]()
  kgen.call @"$mojo::nutty::names[$int::Int]($int::\22Int\22)"<1>() : () -> ()
  // CHECK: call [[F1]]()
  kgen.call @"$mojo::nutty::names[$int::Int]($int::\22Int\22)"<8>() : () -> ()
  // CHECK: call [[F2]]()
  kgen.call @"$no::concrete::plz()"() : () -> ()
  kgen.return
}

// CHECK: kgen.func @_ctor_fn_x_[[CTOR:.*]]()
kgen.generator @"(ctor_fn)x"() {
  // CHECK: kgen.global.address @_some_module_x_[[VAR:.*]] :
  kgen.global.address @"$some_module::x" : <i32>
  kgen.return
}

// CHECK: kgen.func @_dtor_fn_x_[[DTOR:.*]]()
kgen.generator @"(dtor_fn)x"() {
  kgen.return
}

// CHECK: kgen.global @_some_module_x_[[VAR]] : i32 [@_ctor_fn_x_[[CTOR]], @_dtor_fn_x_[[DTOR]]](2)
kgen.global @"$some_module::x" : i32 [@"(ctor_fn)x", @"(dtor_fn)x"](2)
