// RUN: mkdir -p %t.dir && cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP

// CHECK-LABEL: func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"3AaDGffw22m3n1KI45nWbQrhVcILBpo7JmW0cfuu3Ks=">]>}

// ROUNDTRIP: func.func private @trivial(%arg0: i32) {
// ROUNDTRIP:   "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @trivial(%arg0: i32) {
  "some.op"(%arg0) {} : (i32) -> () loc("foo":0:0)
  return loc("foo":1:0)
}

// COM: Empty funcs still have regions, they're just...empty! Their hashes must therefore match.

// CHECK-LABEL: func.func private @empty1() attributes {region_hashes = #cache<regions[<"47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=">]>}
func.func private @empty1()
// CHECK-LABEL: func.func private @empty2() attributes {region_hashes = #cache<regions[<"47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=">]>}
func.func private @empty2()

// CHECK-LABEL: func.func private @caller(i32) attributes {region_hashes = #cache<regions[<"A+5v4ZGYMclDMrA4OL2d+rd28V+ERy9/jLvTnw+jTe0=" symbols = [@trivial]>]>}

// ROUNDTRIP: func.func private @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> () loc("foo":0:0)
  return loc("foo":1:0)
}

// CHECK-LABEL: func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"vHPTJ2UIB98hG+obnXbxm/U7+jkxWTW9KGDQyB7GiXM=" symbols = [@trivial, @caller]>]>}

// ROUNDTRIP: func.func private @multi_caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @multi_caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> () loc("foo":0:0)
  call @caller(%arg0) : (i32) -> () loc("foo":1:0)
  return loc("foo":2:0)
}

func.func private @another_trivial(%arg0: i32) {
  return loc("foo":0:0)
}

// CHECK-LABEL: "some.symbol.op"() ({
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }, {
// CHECK-NEXT:  }) {region0_type = (i32) -> (), region_hashes = #cache<regions[
// CHECK-SAME: <"fGXOPpj0hfczNrKloYp6qFF/dvzv8dUGTNbI6F0eKHA=" symbols = [@trivial]>
// COM: These two are the same because we put in the same locations and they have the same ops.
// CHECK-SAME: <"aMHTL9MhjHJonpIuLQzuEawWvNSs7Mgk9wuvH/sxpu8=" symbols = [@trivial, @caller]>
// CHECK-SAME: <"aMHTL9MhjHJonpIuLQzuEawWvNSs7Mgk9wuvH/sxpu8=" symbols = [@another_trivial, @caller]>
// COM: This one should be different because the cache indices are different.
// CHECK-SAME: <"eiFeLEEcDiPwXwmbwmSwahkHT+Ys6xqr8pH/Y6nscF8=" symbols = [@trivial]>
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
  "some.op"() {} : () -> () loc("foo":0:0)
  func.call @trivial(%arg0) : (i32) -> () loc("foo":1:0)
}, {
^bb0(%arg0: i32):
  "some.op"() {} : () -> () loc("foo":2:0)
  func.call @trivial(%arg0) : (i32) -> () loc("foo":3:0)
  "some.other.op"() {} : () -> () loc("foo":4:0)
  func.call @caller(%arg0) : (i32) -> () loc("foo":5:0)
}, {
^bb0(%arg0: i32):
  "some.op"() {} : () -> () loc("foo":2:0)
  func.call @another_trivial(%arg0) : (i32) -> () loc("foo":3:0)
  "some.other.op"() {} : () -> () loc("foo":4:0)
  func.call @caller(%arg0) : (i32) -> () loc("foo":5:0)
}, {
 ^bb0(%arg0: i32):
   "some.op"() {} : () -> () loc("foo":10:0)
   func.call @trivial(%arg0) : (i32) -> () loc("foo":11:0)
   "some.other.op"() {} : () -> () loc("foo":12:0)
   func.call @trivial(%arg0) : (i32) -> () loc("foo":13:0)
 }) {sym_name = "multi_region", region0_type = (i32) -> ()} : () -> ()
