// RUN: cache-opt -inflate-symbols -split-input-file -verify-diagnostics

// expected-error-re@below {{hash '[a-zA-Z+=/]+' could not be found in the cache}}
func.func private @no_symbols(i32) attributes {region_hashes = #cache<regions[<"KZjGvTfYCCkbb9PgvO+mUGyo+jhy1GVXoPxy+BWmww==">]>}

// -----

func.func private @constant_only() -> tensor<4xf64> {
  // expected-error-re@below {{hash '[a-zA-Z+=/]+' could not be found in the cache}}
  %0 = arith.constant #cache.constant_hash<"4uIDOufhnWgFmdTrChNZorSOxbqsdQZsMX+/hRWcVO==", {align = 8 : ui64, name = "aconstant"} : tensor<4xf64>> : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

// expected-error-re@below {{hash '[a-zA-Z+=/]+' could not be found in the cache}}
func.func private @nested_symbols(i32) attributes {region_hashes = #cache<regions[<"TWsbzGIvYfTNilkZPcjuUuPdm23nP/6TPAzwRp8QCX==" symbols = [@trivial, @caller]>]>}

// -----

// expected-error-re@below {{hash '[a-zA-Z+=/]+' could not be found in the cache}}
func.func private @nested_constant(i32) attributes {region_hashes = #cache<regions[
  <"TsbaqAeb7ZQ8x9+PDp+mPqRyrd3zDBwh13XmtdReoU==" hashes = [
    <"4uIDOufhnWgFmdTrChNZorSOxbqsdQZsMX+/hRWcVO==", {align = 8 : ui64, name = "aconstant"} : tensor<4xf64>>
  ]>]>}
