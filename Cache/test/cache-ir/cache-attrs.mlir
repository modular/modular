// RUN: cache-opt -allow-unregistered-dialect %s | cache-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: no.callees
// CHECK-SAME: cache.region_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="
"no.callees"() {
  attr = #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
} : () -> ()

func.func private @afunc()

// CHECK-LABEL: with.callees
// CHECK-SAME: cache.region_hash
// CHECK-SAME: "YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="[@afunc]
"with.callees"() {
  attr = #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo="[@afunc]>
} : () -> ()

// CHECK-LABEL: symbol.ref
// CHECK-SAME: #cache.symbol_index<123348>
"symbol.ref"() {
  attr = #cache.symbol_index<123348>
} : () -> ()
