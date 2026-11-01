// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.device_graph.collective: block argument #0 must be a device-graph
//   builder.
//

func.func @collective_bad_builder_type() {
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 block argument #0 has type '!mgp.stable_addr<<4xf32, <"gpu", 0>>>'; expected `!mgp.device_graph.builder`}}
  %0 = mgp.device_graph.collective builder(
      %b: !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: every declared slot must be a stable address.
//   The slot list's attribute type admits nothing else, so this is a parse
//   error rather than a verifier diagnostic.
//

func.func @collective_bad_stable_type() {
  %0 = mgp.device_graph.collective builder(
      %b
  ) stable(
      // expected-error@+1 {{invalid kind of type specified: expected mgp.stable_addr, but found '!mgp.tensor<4xf32>'}}
      !mgp.tensor<4xf32>
  ) <"gpu", 0> {
    mgp.device_graph.collective.yield
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: each yield operand type must match the
//   corresponding op result type.
//

func.func @collective_bad_yield_type() {
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 yield operand type '!mgp.stable_addr<<4xf32, <"gpu", 0>>>' does not match output #0 type '!mgp.tensor<4xf32>'}}
  %0 = mgp.device_graph.collective builder(
      %b
  ) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield %a0 : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.collective: a region must not yield more operands than
//   the op declares outputs.
//

func.func @collective_too_many_yields() {
  %val = "test.op"() : () -> !mgp.tensor<4xf32>
  // expected-error@+1 {{'mgp.device_graph.collective' op region #0 yields more operands than declared outputs (1)}}
  %0 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    mgp.device_graph.collective.yield %val, %val : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
  } : !mgp.tensor<4xf32>

  return
}

// -----

//
// mgp.device_graph.stable.get: the slot index must exist in the enclosing
//   collective.
//

func.func @stable_get_index_out_of_range() {
  mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    // expected-error@+1 {{'mgp.device_graph.stable.get' op stable slot index 1 is out of range; the enclosing collective declares 1 slot(s)}}
    %a = mgp.device_graph.stable.get %b[1] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
    mgp.device_graph.collective.yield
  }

  return
}

// -----

//
// mgp.device_graph.stable.get: the result type must be the declared slot type.
//

func.func @stable_get_type_mismatch() {
  mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    // expected-error@+1 {{'mgp.device_graph.stable.get' op stable slot #0 is declared as '!mgp.stable_addr<<4xf32, <"gpu", 0>>>', but the op uses '!mgp.stable_addr<<4xf32, <"gpu", 1>>>'}}
    %a = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    mgp.device_graph.collective.yield
  }

  return
}

// -----

//
// mgp.device_graph.stable.place: the tensor's slot type must be the declared
//   slot type.
//

func.func @stable_place_type_mismatch() {
  %val = "test.op"() : () -> !mgp.tensor<4xf32, <"gpu", 1>>
  mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    // expected-error@+1 {{'mgp.device_graph.stable.place' op stable slot #0 is declared as '!mgp.stable_addr<<4xf32, <"gpu", 0>>>', but the op uses '!mgp.stable_addr<<4xf32, <"gpu", 1>>>'}}
    %p = mgp.device_graph.stable.place %b[0], %val : !mgp.tensor<4xf32, <"gpu", 1>>
    mgp.device_graph.collective.yield
  }

  return
}

// -----

//
// mgp.device_graph.stable.create: input tensor must not already be `stable`.
//

func.func @stable_create_already_stable(%t: !mgp.tensor<4xf32, stable>) -> !mgp.tensor<4xf32, stable> {
  // expected-error@+1 {{'mgp.device_graph.stable.create' op input tensor is already marked `stable`}}
  %s = mgp.device_graph.stable.create %t : !mgp.tensor<4xf32, stable>
  return %s : !mgp.tensor<4xf32, stable>
}

// -----

//
// mgp.device_graph.stable.create: one runtime dim per dynamic dimension.
//

func.func @stable_create_missing_runtime_dim(%t: !mgp.tensor<?x3xf32, <"gpu", 0>>) -> !mgp.tensor<?x3xf32, <"gpu", 0>, stable> {
  // expected-error@+1 {{'mgp.device_graph.stable.create' op expected 1 runtime dimension(s), one per dynamic dimension of the tensor, but got 0}}
  %s = mgp.device_graph.stable.create %t : !mgp.tensor<?x3xf32, <"gpu", 0>>
  return %s : !mgp.tensor<?x3xf32, <"gpu", 0>, stable>
}

// -----

//
// mgp.device_graph.stable.tensor: the spec must be the slot element type's.
// The spec's type is derived from the slot, so a mismatch is a type error at
// the use rather than a verifier diagnostic.
//

func.func @stable_tensor_spec_mismatch(%a: !mgp.stable_addr<<4xf32, <"gpu", 0>>>) {
  // expected-note@+1 {{prior use here}}
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<8xf32>
  // expected-error@+1 {{use of value '%spec' expects different type than prior uses: '!mgp.tensor_spec<4xf32>' vs '!mgp.tensor_spec<8xf32>'}}
  %t = mgp.device_graph.stable.tensor %a, %spec : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  return
}
