// RUN: kgen-opt -decompose-function-arguments -allow-unregistered-dialect -split-input-file -verify-diagnostics %s

// `pop.store` through a decomposed pointer arg is rejected unconditionally,
// even though this test policy's pointer leaf happens to survive intact:
// the pass doesn't track whether a given path still denotes the original
// pointer versus something rebuilt, so it can't tell this case apart from
// one where writing through the "pointer" wouldn't reach real storage.
kgen.func @store_through_decomposed_ptr(%arg0: !kgen.struct<(pointer<struct<(i32)>>)>) {
  %p = kgen.struct.extract %arg0[0] : <(pointer<struct<(i32)>>)>
  %scalar = "produce"() : () -> i32
  %v = kgen.struct.create(%scalar) : !kgen.struct<(i32)>
  // expected-error @+1 {{pop.store against a decomposed pointer arg}}
  pop.store %v, %p : !kgen.pointer<struct<(i32)>>
  hlcf.return
}

// -----

// A leaf is a constant path, so a symbolic index can never match one.
kgen.func @symbolic_index(%arg0: !kgen.struct<(i32, i32)>) {
  // expected-error @+1 {{indexes a decomposed arg with a non-constant index}}
  %0 = kgen.struct.extract %arg0[#kgen.param.decl.ref<"i"> : index] : <(i32, i32)>
  "use"(%0) : (i32) -> ()
  hlcf.return
}
