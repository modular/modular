// RUN: mo-opt %s --split-input-file --verify-diagnostics

// A constant perm is in the spec, so the op takes no permutations operand.
func.func @static_perm_with_operand(%idx: !map.index<2>, %perm: !mo.tensor<[2], si64>) {
  // expected-error @below{{takes a permutations operand only when some entry of the spec's perm is dynamic}}
  %0 = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>>(%idx, %perm)
      : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
  return
}

// -----

// A dynamic perm entry is read from the permutations operand.
func.func @dynamic_perm_without_operand(%idx: !map.index<2>) {
  // expected-error @below{{takes a permutations operand only when some entry of the spec's perm is dynamic}}
  %0 = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [?, ?]>>(%idx)
      : (!map.index<2>) -> !map.index<2>
  return
}
