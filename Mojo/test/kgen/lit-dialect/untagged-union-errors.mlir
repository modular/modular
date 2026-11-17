// 1. Reject invalid union initialization, member indices and result references.
// 2. Reject constant fragments with unsupported types.
// RUN: kgen-opt %s -kgen-verifier -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @initialize_not_a_union(%arg: !kgen.pointer<i8>) {
  // expected-error @below {{operand #0 must be pointer to}}
  kgen.untagged_union.init %arg : !kgen.pointer<i8>
  hlcf.return
}

// -----

kgen.generator @not_a_union(%arg: !lit.ref<i8, mut #lit.any.origin>) {
  // expected-error @below {{container must reference untagged union storage}}
  %member = lit.ref.union.cast %arg[0] : !lit.ref<i8, mut #lit.any.origin> -> !lit.ref<i8, mut #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @bad_index(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) {
  // expected-error @below {{member index out of range}}
  %member = lit.ref.union.cast %arg[1] : !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin> -> !lit.ref<i8, mut #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @negative_index(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) {
  // expected-error @below {{member index must be nonnegative}}
  %member = lit.ref.union.cast %arg[-1] : !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin> -> !lit.ref<i8, mut #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @bad_member(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) {
  // expected-error @below {{result must preserve the member type, container origin and address space}}
  %member = lit.ref.union.cast %arg[0] : !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin> -> !lit.ref<i64, mut #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @bad_origin(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) {
  // expected-error @below {{result must preserve the member type, container origin and address space}}
  %member = lit.ref.union.cast %arg[0] : !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin> -> !lit.ref<i8, imm #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @bad_address_space(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin, 3>) {
  // expected-error @below {{result must preserve the member type, container origin and address space}}
  %member = lit.ref.union.cast %arg[0] : !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin, 3> -> !lit.ref<i8, mut #lit.any.origin>
  hlcf.return
}

// -----

kgen.generator @bad_fragment() {
  // expected-error @below {{union fragments must be bytes, pointers or function symbols}}
  %value = kgen.param.constant: !kgen.untagged_union<i64> = <#kgen.untagged_union<[5 : i64]> : !kgen.untagged_union<i64>>
  hlcf.return
}

// -----

kgen.generator @non_index_member(%arg: !lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) {
  // expected-error @below {{attribute 'index' failed to satisfy constraint}}
  %member = "lit.ref.union.cast"(%arg) {index = 0 : i8} : (!lit.ref<!kgen.untagged_union<i8>, mut #lit.any.origin>) -> !lit.ref<i8, mut #lit.any.origin>
  hlcf.return
}
