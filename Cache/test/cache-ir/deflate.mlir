// RUN: mkdir -p %t.dir && cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP-2

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

// CHECK-LABEL: func.func private @caller(i32) attributes {region_hashes = #cache<regions[<"rjhHwzn5lib4YRnCnsMGONZNPeRh79GPth0d/L+ctwY=" params = [@trivial]>]>}

// ROUNDTRIP: func.func private @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> () loc("foo":0:0)
  return loc("foo":1:0)
}

// CHECK-LABEL: func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"ufYZtAgaPaecuiOem1yKSaTYH5FLOLtXbFTrs+LdWW8=" params = [@trivial, @caller]>]>}

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
// CHECK-SAME: <"iwA8HMUCR1mccyfXxkovhFCcKV1a07wT8kQifWBon3Q=" params = [@trivial]>
// COM: These two are the same because we put in the same locations and they have the same ops.
// CHECK-SAME: <"B9ExCbGaBEMC5mYyvsIRiMGj21UCFOYMm3dvkFEFn28=" params = [@trivial, @caller]>
// CHECK-SAME: <"B9ExCbGaBEMC5mYyvsIRiMGj21UCFOYMm3dvkFEFn28=" params = [@another_trivial, @caller]>
// COM: This one should be different because the cache indices are different.
// CHECK-SAME: <"/crsbAwDWzSlsBVJjgpKzitmtwcKDBXJfr3QPjyM+a8=" params = [@trivial]>
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

// COM: This ensures we can handle nested objects.
// CHECK-LABEL: "some.symbol.op"() ({
// CHECK-NEXT:  }) {region_hashes = #cache<regions[<"9KjmVgntv6hWsCzHXuGnILLqf8pg5vYtCvcD9E8YXZY=" params = [@trivial, @nested::@trivial]>]>, sym_name = "nested"}

// ROUNDTRIP: "some.symbol.op"() ({
// ROUNDTRIP:   func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"3AaDGffw22m3n1KI45nWbQrhVcILBpo7JmW0cfuu3Ks=">]>}
// ROUNDTRIP:   func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"NxHBQYHF2NAdK5Kq+2ao4zG+o6TVxLMPEvWk7mcQSlY=" params = [@trivial, @nested::@trivial]>]>}
// ROUNDTRIP: }) {sym_name = "nested"} : () -> ()

// ROUNDTRIP-2: "some.symbol.op"() ({
// ROUNDTRIP-2:   func.func private @trivial(%arg0: i32) {
// ROUNDTRIP-2:     "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP-2:     return
// ROUNDTRIP-2:   }
// ROUNDTRIP-2:   func.func private @multi_caller(%arg0: i32) {
// ROUNDTRIP-2:     "someop.call"(%arg0) {callee = @nested::@trivial} : (i32) -> ()
// ROUNDTRIP-2:     return
// ROUNDTRIP-2:   }
// ROUNDTRIP-2: }) {sym_name = "nested"} : () -> ()

"some.symbol.op"() ({
  func.func private @trivial(%arg0: i32) {
    "some.op"(%arg0) {} : (i32) -> () loc("foo":0:0)
    return loc("foo":1:0)
  } loc("foo":2:0)

  func.func private @multi_caller(%arg0: i32) {
    "someop.call"(%arg0) {callee = @nested::@trivial} : (i32) -> () loc("foo":3:0)
    return loc("foo":4:0)
  } loc("foo":5:0)
}) {sym_name = "nested"} : () -> ()
