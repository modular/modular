// RUN: mkdir -p %t.dir && cache-opt -deflate-constants="cache-dir=%t.dir" %s | FileCheck %s
// RUN: cache-opt -deflate-constants="cache-dir=%t.dir" %s | cache-opt -inflate-constants="cache-dir=%t.dir" | FileCheck %s -check-prefix=ROUNDTRIP
// RUN: cache-opt -deflate-constants="cache-dir=%t.dir" -deflate-symbols="cache-dir=%t.dir" %s | FileCheck %s -check-prefix=NESTED

// CHECK-LABEL: @trivial
// CHECK: #cache.constant_hash<"ABpMw6fFxznfdZ3ywFY8giTVyom+f7q9mc9WiNKgSRU=", {align = 8 : ui64, name = "aconstant"} : tensor<4xf64>>
// CHECK-NEXT: call
// CHECK-NEXT: return

// ROUNDTRIP-LABEL: @trivial
// ROUNDTRIP: dense_resource<aconstant> : tensor<4xf64>
// ROUNDTRIP: dialect_resources
// ROUNDTRIP: builtin
// ROUNDTRIP: aconstant: "0x08000000010000000000000002000000000000000300000000000000"

// NESTED-LABEL: @trivial
// NESTED-SAME: attributes {region_hashes = #cache<regions[
// COM: First the hash of the region itself.
// NESTED-SAME:   "2XLu45AyYLctXxee19F/QSQ+/5snWmsdBRdJ+GEyDA0=" params =
// COM: Next, the hashes inside (from the deflated constant).
// NESTED-SAME:   #cache.constant_hash<"ABpMw6fFxznfdZ3ywFY8giTVyom+f7q9mc9WiNKgSRU=", {align = 8 : ui64, name = "aconstant"} : tensor<4xf64>>
// COM: Next, the symbols referred-to inside the region
// NESTED-SAME:   @external

func.func private @external()

func.func private @trivial() -> tensor<4xf64> {
  %0 = arith.constant dense_resource<aconstant> : tensor<4xf64> loc("foo":0:0)
  call @external() : () -> () loc("foo":1:0)
  return %0 : tensor<4xf64> loc("foo":2:0)
}

{-#
  dialect_resources: {
    builtin: {
      aconstant: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}
