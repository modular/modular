// RUN: cache-opt -allow-unregistered-dialect %s | cache-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: no.align
// CHECK-SAME: constant_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="
// CHECK-SAME: i32
"no.align"() {
  attr = #cache.constant_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=" : tensor<1xi32>>
} : () -> ()

// CHECK-LABEL: with.align
// CHECK-SAME: constant_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="
// CHECK-SAME: align = 1234 : i32
// CHECK-SAME: i32
"with.align"() {
  attr = #cache.constant_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=", {align=1234:i32} : tensor<1xi32>>
} : () -> ()

// CHECK-LABEL: no.callees
// CHECK-SAME: cache.region_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="
"no.callees"() {
  attr = #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
} : () -> ()

func.func private @afunc()

// CHECK-LABEL: with.callees
// CHECK-SAME: cache.region_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=" params = [@afunc]
"with.callees"() {
  attr = #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=" params=[@afunc]>
} : () -> ()

// CHECK-LABEL: with.callees
// CHECK-SAME: cache.region_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=" params = [@afunc,
// CHECK-SAME: #cache.constant_hash<"ZTA5YjE2ODExNDQ0NDAxYjM1Yzk0MDgxZWU4YzgyYTc2MWJjZDNjZmQ3MjYwY2YwNjNlM2ZlYzUyMGY1ZjVlOQo=" : tensor<1234x1234xf32>> : tensor<1234x1234xf32>]>
"with.callees.and.hashes"() {
  attr = #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="
                            params=[@afunc, #cache.constant_hash<"ZTA5YjE2ODExNDQ0NDAxYjM1Yzk0MDgxZWU4YzgyYTc2MWJjZDNjZmQ3MjYwY2YwNjNlM2ZlYzUyMGY1ZjVlOQo=" : tensor<1234x1234xf32>>]>
} : () -> ()

// CHECK-LABEL: hash.ref
// CHECK-SAME: #cache.hash_index<12338>
"hash.ref"() {
  attr = #cache.hash_index<12338>
} : () -> ()
