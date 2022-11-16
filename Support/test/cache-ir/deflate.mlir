// RUN: mkdir -p %t.dir && support-dialect-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: support-dialect-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | support-dialect-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP

// CHECK-LABEL: cache.symbol @trivial
// CHECK-SAME: "func.func"
// CHECK-SAME: #cache.region_hash<"Helo8YTV6la3ppYsF027qaKo66t5ZZ5CVZZ+zwJZdPY=">
// CHECK-SAME: original_attrs={function_type = (i32) -> ()}

// ROUNDTRIP: func.func @trivial(%arg0: i32) {
// ROUNDTRIP:   "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func @trivial(%arg0: i32) {
  "some.op"(%arg0) {} : (i32) -> ()
  return
}

// CHECK-LABEL: cache.symbol @caller
// CHECK-SAME: "func.func"
// CHECK-SAME: #cache.region_hash<"qwHWcM+AjoNE3ikRZ6dloFvfZUFWNim7YrTM9Wa6Nbs="[@trivial]>
// CHECK-SAME: original_attrs={function_type = (i32) -> ()}

// ROUNDTRIP: func.func @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: cache.symbol @multi_caller
// CHECK-SAME: "func.func"
// CHECK-SAME: #cache.region_hash<"q8djFUJeW9vLRSzOHa6m0rjExpA8PByroiMngjBBSHU="[@trivial, @caller]>

// ROUNDTRIP: func.func @multi_caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func @multi_caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> ()
  call @caller(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: cache.symbol @multi_region
// CHECK-SAME: "some.symbol.op"
// CHECK-SAME: #cache.region_hash<"vYK+gqhtWWNKkQBRppXGIz6Of6v8IKMCgi+Ir351ISI="[@trivial]>
// COM: We have 2 of the same region, so their hashes should match.
// CHECK-SAME: #cache.region_hash<"+ocDsdOifBRT7FgyNUpodhdj1Nhf6Ws6rU1fCjTZhNo="[@trivial, @caller]>
// CHECK-SAME: #cache.region_hash<"+ocDsdOifBRT7FgyNUpodhdj1Nhf6Ws6rU1fCjTZhNo="[@trivial, @caller]>
// CHECK-SAME: original_attrs={region0_type = (i32) -> ()}

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
// ROUNDTRIP:   func.call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   "some.other.op"() : () -> ()
// ROUNDTRIP:   func.call @caller(%arg0) : (i32) -> ()
// ROUNDTRIP: }) {region0_type = (i32) -> (), sym_name = "multi_region"} : () -> ()

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
  func.call @trivial(%arg0) : (i32) -> ()
  "some.other.op"() {} : () -> ()
  func.call @caller(%arg0) : (i32) -> ()
}) {sym_name = "multi_region", region0_type = (i32) -> ()} : () -> ()
