// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-livm, map.iter.opaque(canonicalize,cse)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.transpose chain feeding a
// map.load out of the loop body into a map.index.transpose.stride op in the
// enclosing map.iter.opaque, rewriting the load to use the loop's own
// (untransformed) index directly -- same generic IndexTransformOpInterface
// mechanism already exercised by map.index.slice/map.index.broadcast, no
// pass changes needed. The transpose's perm operand rides along into the
// hoisted stride op (like slice's bounds), and canonicalize then drops the
// now-dead original input capture.

module {

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[2, 3], f32>, [[PERM:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 2], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.transpose.stride<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> ([[INPUT]], [[PERM]])
// CHECK-NEXT:      map.iter.elementwise.body<[3, 2]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[3, 2], f32>, [[NEWCAP:%[^ ]+]] = [[STRIDED]] : !mo.tensor<[3, 2], f32>} {
// CHECK-NOT:       map.index.transpose
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[NEWCAP]]{{\[}}[[IDX]]] : !mo.tensor<[3, 2], f32>
// CHECK-NEXT:      map.store [[OUT0]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 2], f32>
// CHECK-NEXT:    }
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>, %perm: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 2], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[2, 3], f32>, %perm : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 2], f32>) {
    map.iter.elementwise.body<[3, 2]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[2, 3], f32>, %perm0 = %perm : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 2], f32>} {
      %idx_ = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> (%idx, %perm0)
          : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input0[%idx_] : !mo.tensor<[2, 3], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 2], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 2], f32>
}

// 3D permutation: out.dim[i] = in.dim[perm[i]] for perm = [2, 0, 1].

// CHECK-LABEL: mo.graph @transpose3D
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT2:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>, [[PERM2:%[^ :,]+]]: !mo.tensor<[3], si64>)
// CHECK-NEXT:    outputs ([[OUT2:%[^ :,]+]] : !mo.tensor<[4, 2, 3], f32>) {
// CHECK-NEXT:      [[STRIDED2:%[^ ]+]] = map.index.transpose.stride<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> ([[INPUT2]], [[PERM2]])
// CHECK-NEXT:      map.iter.elementwise.body<[4, 2, 3]>
// CHECK-NEXT:        index   ([[IDX2:%[^ :,]+]]: !map.index<3>)
// CHECK-NEXT:        capturing {{{.*}}[[OUT02:%[^ ,]+]] = [[OUT2]] : !mo.tensor<[4, 2, 3], f32>, [[NEWCAP2:%[^ ]+]] = [[STRIDED2]] : !mo.tensor<[4, 2, 3], f32>} {
// CHECK-NOT:       map.index.transpose
// CHECK-NEXT:      [[VAL2:%[^ ]+]] = map.load [[NEWCAP2]]{{\[}}[[IDX2]]] : !mo.tensor<[4, 2, 3], f32>
// CHECK-NEXT:      map.store [[OUT02]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 2, 3], f32>
// CHECK-NEXT:    }
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>, %perm: !mo.tensor<[3], si64>)
    -> !mo.tensor<[4, 2, 3], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[2, 3, 4], f32>, %perm : !mo.tensor<[3], si64>)
      outputs (%out : !mo.tensor<[4, 2, 3], f32>) {
    map.iter.elementwise.body<[4, 2, 3]>
        index (%idx: !map.index<3>)
        capturing {%input0 = %input : !mo.tensor<[2, 3, 4], f32>, %perm0 = %perm : !mo.tensor<[3], si64>, %out0 = %out : !mo.tensor<[4, 2, 3], f32>} {
      %idx_ = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> (%idx, %perm0)
          : (!map.index<3>, !mo.tensor<[3], si64>) -> !map.index<3>
      %val = map.load %input0[%idx_] : !mo.tensor<[2, 3, 4], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[4, 2, 3], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[4, 2, 3], f32>
}

}
