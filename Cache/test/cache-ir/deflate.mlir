// RUN: mkdir -p %t.dir && cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP
// RUN: cache-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | cache-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP-2

// CHECK-LABEL: func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"llBkdikIv7EAcB6tdsv0qwy6anFSI0GnU/tXwg7z9tc=">]>}

// ROUNDTRIP: func.func private @trivial(%arg0: i32) {
// ROUNDTRIP:   "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @trivial(%arg0: i32) {
  "some.op"(%arg0) {} : (i32) -> () loc("foo":0:0)
  return loc("foo":1:0)
}

// COM: Empty funcs still have regions, they're just...empty! Their hashes must therefore match.

// CHECK-LABEL: func.func private @empty1() attributes {region_hashes = #cache<regions[<"rxNJufX5oaagQE3qNtzJSZvLJcmtwRK3zJqTyuQfMmI=">]>}
func.func private @empty1()
// CHECK-LABEL: func.func private @empty2() attributes {region_hashes = #cache<regions[<"rxNJufX5oaagQE3qNtzJSZvLJcmtwRK3zJqTyuQfMmI=">]>}
func.func private @empty2()

// CHECK-LABEL: func.func private @caller(i32) attributes {region_hashes = #cache<regions[<"Lv7zugCtr0pbO7lcUVHBxx33qmh0RB+d15J61PQotQs=" params = [@trivial]>]>}

// ROUNDTRIP: func.func private @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> () loc("foo":0:0)
  return loc("foo":1:0)
}

// CHECK-LABEL: func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"F8cNWdvbddcfN9620/ytJZo+d+5VTOk3gfr21g40rJk=" params = [@trivial, @caller]>]>}

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
// CHECK-SAME: <"lNPNcnEe0mp8pLiiDJuJ927F9Lnl19U+ijv/qyO0u1o=" params = [@trivial]>
// COM: These two are the same because we put in the same locations and they have the same ops.
// CHECK-SAME: <"20OzxJLkKJs6M7U8IAzjdSPD6cwZ84Jdy2Z5Q7E9lVQ=" params = [@trivial, @caller]>
// CHECK-SAME: <"20OzxJLkKJs6M7U8IAzjdSPD6cwZ84Jdy2Z5Q7E9lVQ=" params = [@another_trivial, @caller]>
// COM: This one should be different because the cache indices are different.
// CHECK-SAME: <"d8ak7dRkauYhd7bZgjCDwoaTrw08vovK8kAJhH9+7pw=" params = [@trivial]>
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
// CHECK-NEXT:  }) {region_hashes = #cache<regions[<"gYrGQAMmy2qKLaaGL8Ti788XoKGsU0VmDKV3QNWrc9Y=" params = [@trivial, @nested::@trivial]>]>, sym_name = "nested"}

// ROUNDTRIP: "some.symbol.op"() ({
// ROUNDTRIP:   func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"llBkdikIv7EAcB6tdsv0qwy6anFSI0GnU/tXwg7z9tc=">]>}
// ROUNDTRIP:   func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"OBlskZEi0vQWY0cRVPvmZ3+UCEehQreRCap+sod0/TA=" params = [@trivial, @nested::@trivial]>]>}
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
