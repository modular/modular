// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Roundtrip tests for the MAP dialect ops and the `!map.index` type. These
// exercise custom parse/print (map.iter.opaque, map.iter.epilogue) and
// declarative formats (map.call.opaque, map.store), so re-running through mo-opt
// (the second RUN line) also checks idempotence.

// map.call.opaque verifies its callee resolves to a mogg.kernel.decl, so the
// decls the kernels below reference are declared here (the import pass cannot be
// scoped to a single kernel, and importing a whole package is far too much for a
// parse test).
mogg.kernel.decl @kernels.matmul registration = "mo.matmul" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// The `!map.index<N>` type prints and parses with its static rank.

// CHECK-LABEL: func.func @index_type_roundtrip
// CHECK-SAME: (%{{.*}}: !map.index<3>) -> !map.index<3>
func.func @index_type_roundtrip(%i: !map.index<3>) -> !map.index<3> {
  return %i : !map.index<3>
}

// A matmul-shaped opaque kernel: two inputs, one rank-2 output, a store
// epilogue that captures the output, a call carrying bracketed params, and a
// return terminator.

// CHECK-LABEL: func.func @opaque_matmul
// CHECK: %[[MM:.*]] = map.iter.opaque
// CHECK: inputs (%[[A:.*]]: !mo.tensor<[128, 256], f32>, %[[B:.*]]: !mo.tensor<[256, 512], f32>)
// CHECK: outputs (%[[OUT:.*]] : !mo.tensor<[128, 512], f32>) {
// CHECK:   %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>) capturing {%[[CAP:.*]] = %[[OUT]] : !mo.tensor<[128, 512], f32>} {
// CHECK:     map.store %[[CAP]][%[[IDX]]] = %[[VAL]] : !mo.tensor<[128, 512], f32>
// CHECK:   } : !mo.tensor<[128, 512], f32>
// CHECK:   map.call.opaque @kernels.matmul[packed_b = false, transpose_b = false] inputs(%[[A]], %[[B]]) outputs(%[[EPI]]) kernelName = "mo.matmul" : (!mo.tensor<[128, 256], f32>, !mo.tensor<[256, 512], f32>) -> !mo.tensor<[128, 512], f32>
// CHECK:   map.yield %[[OUT]] : !mo.tensor<[128, 512], f32>
// CHECK: }
// CHECK: return %[[MM]]
func.func @opaque_matmul(%A: !mo.tensor<[128, 256], f32>, %B: !mo.tensor<[256, 512], f32>) -> !mo.tensor<[128, 512], f32> {
  %mm = map.iter.opaque inputs (%A : !mo.tensor<[128, 256], f32>, %B : !mo.tensor<[256, 512], f32>) outputs (%out : !mo.tensor<[128, 512], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%cap = %out : !mo.tensor<[128, 512], f32>} {
      map.store %cap[%idx] = %val : !mo.tensor<[128, 512], f32>
    } : !mo.tensor<[128, 512], f32>
    map.call.opaque @kernels.matmul[packed_b = false, transpose_b = false] inputs (%A, %B) outputs (%epi) kernelName = "mo.matmul" : (!mo.tensor<[128, 256], f32>, !mo.tensor<[256, 512], f32>) -> (!mo.tensor<[128, 512], f32>)
    map.yield %out : !mo.tensor<[128, 512], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %mm : !mo.tensor<[128, 512], f32>
}

// A unary rank-3 opaque kernel: one input, one output, an empty params list.
// Exercises `!map.index<3>` and the single-input / empty-params paths.

// CHECK-LABEL: func.func @opaque_unary
// CHECK: map.iter.opaque
// CHECK: inputs (%[[X:.*]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 3, 4], f32>) {
// CHECK:   map.iter.epilogue (%[[V:.*]] : f32, %[[I:.*]] : !map.index<3>) capturing {%[[C:.*]] = %[[O]] : !mo.tensor<[2, 3, 4], f32>} {
// CHECK:     map.store %[[C]][%[[I]]] = %[[V]] : !mo.tensor<[2, 3, 4], f32>
// CHECK:   } : !mo.tensor<[2, 3, 4], f32>
// CHECK:   map.call.opaque @kernels.relu[] inputs(%[[X]]) outputs(%{{.*}}) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32>
// CHECK:   map.yield %[[O]] : !mo.tensor<[2, 3, 4], f32>
func.func @opaque_unary(%X: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<3>) capturing {%cap = %out : !mo.tensor<[2, 3, 4], f32>} {
      map.store %cap[%idx] = %val : !mo.tensor<[2, 3, 4], f32>
    } : !mo.tensor<[2, 3, 4], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
    map.yield %out : !mo.tensor<[2, 3, 4], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y : !mo.tensor<[2, 3, 4], f32>
}

// A DPS-only opaque kernel: the output block argument is the result, with no
// `map.yield` terminator at all (the form MOToMAP now emits).

// CHECK-LABEL: func.func @opaque_no_yield
// CHECK: map.iter.opaque
// CHECK: inputs (%[[X:.*]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 3, 4], f32>) {
// CHECK:   map.call.opaque @kernels.relu[] inputs(%[[X]]) outputs(%[[O]]) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32>
// CHECK-NOT: map.yield
// CHECK: }
func.func @opaque_no_yield(%X: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  return %y : !mo.tensor<[2, 3, 4], f32>
}

// A unary rank-3 opaque kernel with both a load prologue on its input and a
// store epilogue on its output. Exercises map.iter.prologue.

// CHECK-LABEL: func.func @opaque_prologue_and_epilogue
// CHECK: map.iter.opaque
// CHECK: inputs (%[[X:.*]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK: outputs (%[[O:.*]] : !mo.tensor<[2, 3, 4], f32>) {
// CHECK:   %[[PRO:.*]] = map.iter.prologue (%[[PI:.*]] : !map.index<3>) capturing {%[[PC:.*]] = %[[X]] : !mo.tensor<[2, 3, 4], f32>} {
// CHECK:     %[[PV:.*]] = map.load %[[PC]][%[[PI]]] : !mo.tensor<[2, 3, 4], f32>
// CHECK:     map.yield %[[PV]] : f32
// CHECK:   } : !mo.tensor<[2, 3, 4], f32>
// CHECK:   %[[EPI:.*]] = map.iter.epilogue (%[[V:.*]] : f32, %[[I:.*]] : !map.index<3>) capturing {%[[C:.*]] = %[[O]] : !mo.tensor<[2, 3, 4], f32>} {
// CHECK:     map.store %[[C]][%[[I]]] = %[[V]] : !mo.tensor<[2, 3, 4], f32>
// CHECK:   } : !mo.tensor<[2, 3, 4], f32>
// CHECK:   map.call.opaque @kernels.relu[] inputs(%[[PRO]]) outputs(%[[EPI]]) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32>
// CHECK:   map.yield %[[O]] : !mo.tensor<[2, 3, 4], f32>
func.func @opaque_prologue_and_epilogue(%X: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %pro = map.iter.prologue (%pidx : !map.index<3>) capturing {%pcap = %X : !mo.tensor<[2, 3, 4], f32>} {
      %pval = map.load %pcap[%pidx] : !mo.tensor<[2, 3, 4], f32>
      map.yield %pval : f32
    } : !mo.tensor<[2, 3, 4], f32>
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<3>) capturing {%cap = %out : !mo.tensor<[2, 3, 4], f32>} {
      map.store %cap[%idx] = %val : !mo.tensor<[2, 3, 4], f32>
    } : !mo.tensor<[2, 3, 4], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
    map.yield %out : !mo.tensor<[2, 3, 4], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y : !mo.tensor<[2, 3, 4], f32>
}
