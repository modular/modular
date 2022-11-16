// RUN: mkdir -p %t.dir && support-dialect-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | FileCheck %s
// RUN: support-dialect-opt -deflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect %s | support-dialect-opt -inflate-symbols="cache-dir=%t.dir" -allow-unregistered-dialect | FileCheck %s -check-prefix=ROUNDTRIP

// CHECK-LABEL: func.func private @trivial(i32) attributes {region_hashes = #cache<regions[<"KZjGvTfYCCkbb9PgvO+mUGyo+jhy1GVXoPxy+BWmwwc=">]>}

// ROUNDTRIP: func.func private @trivial(%arg0: i32) {
// ROUNDTRIP:   "some.op"(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @trivial(%arg0: i32) {
  "some.op"(%arg0) {} : (i32) -> ()
  return
}

// CHECK-LABEL: func.func private @caller(i32) attributes {region_hashes = #cache<regions[<"IrA5UZnO267JoloA9sG21RXfNT0kcH2mYYwUxBr8ESU="[@trivial]>]>}

// ROUNDTRIP: func.func private @caller(%arg0: i32) {
// ROUNDTRIP:   call @trivial(%arg0) : (i32) -> ()
// ROUNDTRIP:   return
// ROUNDTRIP: }

func.func private @caller(%arg0: i32) {
  call @trivial(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL: func.func private @multi_caller(i32) attributes {region_hashes = #cache<regions[<"TWsbzGIvYfTNilkZPcjuUuPdm23nP/6TPAzwRp8QCXE="[@trivial, @caller]>]>}

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
// CHECK-SAME: <"N5Kv1S+GQXt0lF5dDFGNeUInHOHLyUkRuBn/DZ+85v8="[@trivial]>
// COM: thse two *should* be the same - different symbol names but in the same position.
// COM: They aren't because we currently hash locations which are not region-relative.
// CHECK-SAME: <"nnzb/p3DOh4m7lCDNow2CZIehKOfOZEfcqhfOI+1JVw="[@trivial, @caller]>
// CHECK-SAME: <"xaR5TkYNDnHvkXFWksmuRcDHZ9nMQUpmUSV1L8z/R/k="[@another_trivial, @caller]>
// COM: This one should be different because the cache indices are different.
// CHECK-SAME: <"/1An198ZqlfpdZDZIzQ9xptBeJBjpRhcmf2lHskP6Ao="[@trivial]>
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
