// RUN: mkdir -p %t.dir && cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP

// CHECK-LABEL: func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"KZjGvTfYCCkbb9PgvO+mUGyo+jhy1GVXoPxy+BWmwwc=">]>}

// ROUNDTRIP: func.func private @trivial(%arg0: i32) {
// ROUNDTRIP:   "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @trivial(%arg0: i32) {
  "some.op"(%arg0) {} : (i32) -> ()
  return
}

// COM: Empty funcs still have regions, they're just...empty! Their hashes must therefore match.

// CHECK-LABEL: func.func private @empty1() attributes {region_hashes = #cache<regions[<"47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=">]>}
func.func private @empty1()
// CHECK-LABEL: func.func private @empty2() attributes {region_hashes = #cache<regions[<"47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=">]>}
func.func private @empty2()

// CHECK-LABEL: func.func private @caller(i32) attributes {region_hashes = #cache<regions[<"I5l3gqO4/9OLXplvOlYzfapo7ZkT5cSPWQNPBaHDTuM=" symbols = [@trivial]>]>}

// ROUNDTRIP: func.func private @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"TqvkgwX3ENTrquxZn8d/fFBYk9xcQ3UkNVCO/QRoDyg=" symbols = [@trivial, @caller]>]>}

// ROUNDTRIP: func.func private @multi_caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @multi_caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> ()
  call @caller(%arg0) : (i32) -> ()
  return
}

func.func private @another_trivial(%arg0: i32) {
  return
}

// CHECK-LABEL: "some.symbol.op"() ({
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }) {region0_type = (i32) -> (), region_hashes = #cache<regions[
// CHECK-SAME: <"rdBvD5Aa6w7ld/MUW6aFwm4R5UAW/rwWmhVaNLwaCZk=" symbols = [@trivial]>
// COM: thse two *should* be the same - different symbol names but in the same position.
// COM: They aren't because we currently hash locations which are not region-relative.
// CHECK-SAME: <"xYXGZG7b0AFAOhX61T9C8DHDL8nd5FzbnC7sg6cOFc8=" symbols = [@trivial, @caller]>
// CHECK-SAME: <"WzI/hr1vwJKONW/7I7fXUTOBMbj90PTPR8flZ0v4Kfw=" symbols = [@another_trivial, @caller]>
// COM: This one should be different because the cache indices are different.
// CHECK-SAME: <"waCg8krXr+GAilrjEfOfrHcBieYn10TIfBbeS/R0R9g=" symbols = [@trivial]>
// CHECK-SAME: sym_name = "multi_region"} : () -> ()

// ROUNDTRIP: "some.symbol.op"() ({
// ROUNDTRIP: ^bb0(%arg0: i32):
// ROUNDTRIP:   "some.op"() : () -> ()
// ROUNDTRIP:   func.call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP: }, {
// ROUNDTRIP: ^bb0(%arg0: i32):
// ROUNDTRIP:   "some.op"() : () -> ()
// ROUNDTRIP:   func.call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   "some.other.op"() : () -> ()
// ROUNDTRIP:   func.call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP: }, {
// ROUNDTRIP: ^bb0(%arg0: i32):
// ROUNDTRIP:   "some.op"() : () -> ()
// ROUNDTRIP:   func.call @another_trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   "some.other.op"() : () -> ()
// ROUNDTRIP:   func.call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP: }, {
// ROUNDTRIP:  ^bb0(%arg0: i32):
// ROUNDTRIP:    "some.op"() : () -> ()
// ROUNDTRIP:    func.call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:    "some.other.op"() : () -> ()
// ROUNDTRIP:    func.call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:  }) {region0_type = (i32) -> (), sym_name = "multi_region"} : () -> ()

"some.symbol.op"() ({
^bb0(%arg0: i32):
  "some.op"() {} : () -> ()
  func.call @trivial(%arg0) : (i32) -> ()
}, {
^bb0(%arg0: i32):
  "some.op"() {} : () -> ()
  func.call @trivial(%arg0) : (i32) -> ()
  "some.other.op"() {} : () -> ()
  func.call @caller(%arg0) : (i32) -> ()
}, {
^bb0(%arg0: i32):
  "some.op"() {} : () -> ()
  func.call @another_trivial(%arg0) : (i32) -> ()
  "some.other.op"() {} : () -> ()
  func.call @caller(%arg0) : (i32) -> ()
}, {
 ^bb0(%arg0: i32):
   "some.op"() {} : () -> ()
   func.call @trivial(%arg0) : (i32) -> ()
   "some.other.op"() {} : () -> ()
   func.call @trivial(%arg0) : (i32) -> ()
 }) {sym_name = "multi_region", region0_type = (i32) -> ()} : () -> ()
