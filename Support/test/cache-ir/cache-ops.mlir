// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// COM: This ensures we can roundtrip a `cache.symbol`.
// CHECK-LABEL: cache.symbol @afunc
// CHECK-SAME: "func.func" regions=[
// CHECK-SAME:   #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
// CHECK-SAME: ] original_attrs={function_type = () -> ()}
cache.symbol @afunc "func.func"
regions=[
  #cache.region_hash<"YWI1MzBhMTNlNDU5MTQ5ODJiNzlmOWI3ZTNmYmE5OTRjZmQxZjNmYjIyZjcxY2VhMWFmYmYwMmI0NjBjNmQxZAo=">
] original_attrs = {function_type=()->()}

// COM: This ensures that we can roundtrip a `cache.container`.
// CHECK-LABEL: cache.container
// CHECK-NEXT-2: "some.op"
cache.container {
  "some.op"() {} : () -> ()
  "some.op"() {} : () -> ()
}

// COM: This ensures that we can roundtrip a `cache.container`.
// CHECK-LABEL: cache.container attributes {anAttr = "somestring"}
// CHECK-NEXT-2: "some.op"
cache.container attributes {anAttr = "somestring"} {
  "some.op"() {} : () -> ()
  "some.op"() {} : () -> ()
}

// COM: This ensures that we can roundtrip a `cache.container` with block args.
// CHECK-LABEL: cache.container attributes {anAttr = "somestring"}
// CHECK-NEXT: ^bb0(%arg0: i32):
// CHECK-NEXT-2: "some.op"
cache.container attributes {anAttr = "somestring"} {
^bb0(%arg0: i32):
  "some.op"() {} : () -> ()
  "some.op"() {} : () -> ()
}
