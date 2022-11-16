// RUN: cache-opt -allow-unregistered-dialect %s | cache-opt -allow-unregistered-dialect | FileCheck %s

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
