// RUN: mo-opt %s -split-input-file -verify-diagnostics

// mogg.routine verifier: the body block arguments are one !mogg.pointer per
// input then output tensor, then zero or more !mogg.dim, then exactly one
// trailing !mogg.context. The labeled groups (inputs/outputs/context/dims)
// reconstruct that in-memory order regardless of display order. Each op carries
// the `device` attribute required by the `MOExplicitDevice` trait so these cases
// exercise the body-block-arg verifier rather than the missing-device check.

// Too few block args: 1 input + 1 output = 2 pointers + a context (3).
func.func @too_few(%a: !mo.tensor<[2, 2], f32>) {
  // expected-error @+1 {{expected at least 3 body block arguments}}
  %k = mogg.routine (%a : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs(%p : !mogg.pointer)
    context(%ctx : !mogg.context) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.call.opaque @noop
  }
  return
}

// -----

// A non-pointer where a per-tensor pointer block arg is required (a !mogg.dim
// leads the inputs group, so body block arg #0 is not a pointer).
func.func @not_pointer(%a: !mo.tensor<[2, 2], f32>) {
  // expected-error @+1 {{body block argument #0 must be a !mogg.pointer}}
  %k = mogg.routine (%a : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs(%d : !mogg.dim, %p : !mogg.pointer)
    context(%ctx : !mogg.context) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.call.opaque @noop
  }
  return
}

// -----

// Trailing context missing (a stray dim where the context must be).
func.func @no_context(%a: !mo.tensor<[2, 2], f32>) {
  // expected-error @+1 {{must be zero or more !mogg.dim followed by one or more !mogg.context}}
  %k = mogg.routine (%a : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs(%p0 : !mogg.pointer, %p1 : !mogg.pointer)
    dims(%d : !mogg.dim) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.call.opaque @noop
  }
  return
}
