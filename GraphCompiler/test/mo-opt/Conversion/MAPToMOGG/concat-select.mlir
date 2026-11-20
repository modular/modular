// RUN: mo-opt %s --mogg-import-packages --map-to-mogg | FileCheck %s

// A foreach reading through a map.index.concat.select lowers to a mogg.call.foreach holding a
// mogg.index.concat.select: one region per input, its map.yield a mogg.yield.
// The select carries every input's extent on the axis and whether the axis is
// the innermost one; when an extent is dynamic, it also takes every input's
// runtime shape, built in kernel scope.

// CHECK-LABEL: mo.graph @concatStatic
// CHECK:       mogg.call.foreach @{{.*}}._foreach shape(%{{.*}}) context(%{{.*}}) (%[[IDX:.*]] : !map.index<2>) capturing {%[[A:.*]] = %{{.*}} : !mogg.tensor, %[[B:.*]] = %{{.*}} : !mogg.tensor, %[[OUT:.*]] = %{{.*}} : !mogg.tensor}
// CHECK-NEXT:    %[[V:.*]] = mogg.index.concat.select(%[[IDX]]) <axis = 1, sizes = #mosh<ape[8, 16]> : !mosh.ape> : !map.index<2> -> !mogg.simd<f32> {
// CHECK-NEXT:    ^bb0(%[[I0:.*]]: !map.index<2>):
// CHECK-NEXT:      %[[X0:.*]] = mogg.tensor.load %[[A]][%[[I0]]]
// CHECK-NEXT:      mogg.yield %[[X0]] : !mogg.simd<f32>
// CHECK-NEXT:    }, {
// CHECK-NEXT:    ^bb0(%[[I1:.*]]: !map.index<2>):
// CHECK-NEXT:      %[[X1:.*]] = mogg.tensor.load %[[B]][%[[I1]]]
// CHECK-NEXT:      mogg.yield %[[X1]] : !mogg.simd<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    mogg.tensor.store %[[OUT]][%[[IDX]]] = %[[V]]
mo.graph @concatStatic(%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[4, 16], f32>)
    outputs (%out : !mo.tensor<[4, 24], f32>) {
    map.iter.elementwise.body<[4, 24]> max_simd_width(8)
      index   (%arg4: !map.index<2>)
      capturing {%arg5 = %arg0 : !mo.tensor<[4, 8], f32>, %arg6 = %arg1 : !mo.tensor<[4, 16], f32>, %arg7 = %out : !mo.tensor<[4, 24], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%arg4) : !map.index<2> -> f32 (%arg8: !map.index<2>) {
        %2 = map.load %arg5[%arg8] : !mo.tensor<[4, 8], f32>
        map.yield %2 : f32
      }, (%arg8: !map.index<2>) {
        %2 = map.load %arg6[%arg8] : !mo.tensor<[4, 16], f32>
        map.yield %2 : f32
      }
      map.store %arg7[%arg4] = %1 : !mo.tensor<[4, 24], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[4, 24], f32>
}

// CHECK-LABEL: mo.graph @concatDynamic
// CHECK:       dims (%[[M:.*]]: !mogg.dim<M>, %[[N0:.*]]: !mogg.dim<N0>, %[[N1:.*]]: !mogg.dim<N1>, %{{.*}}: !mogg.dim<P>)
// CHECK:       mogg.tensor.create({{.*}}io = Output
// CHECK:       %[[S0:.*]] = mogg.shape.create(%[[M]], %[[N0]])
// CHECK-NEXT:  %[[S1:.*]] = mogg.shape.create(%[[M]], %[[N1]])
// CHECK:       mogg.call.foreach {{.*}} dynamic_max_simd_width(
// CHECK-NEXT:    mogg.index.concat.select(%{{.*}}, %[[S0]], %[[S1]]) <axis = 1, sizes = #mosh<ape[N0, N1]> : !mosh.ape>
mo.graph @concatDynamic<M, N0, N1, P>(%arg0: !mo.tensor<[M, N0], f32>, %arg1: !mo.tensor<[M, N1], f32>) -> !mo.tensor<[M, P], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[M, N0], f32>, %arg1: !mo.tensor<[M, N1], f32>)
    outputs (%out : !mo.tensor<[M, P], f32>) {
    map.iter.elementwise.body<[M, P]> max_simd_width(#map.gcd<[N0, N1]>)
      index   (%arg4: !map.index<2>)
      capturing {%arg5 = %arg0 : !mo.tensor<[M, N0], f32>, %arg6 = %arg1 : !mo.tensor<[M, N1], f32>, %arg7 = %out : !mo.tensor<[M, P], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[M, N0], [M, N1] -> [M, P], axis = 1>> (%arg4) : !map.index<2> -> f32 (%arg8: !map.index<2>) {
        %2 = map.load %arg5[%arg8] : !mo.tensor<[M, N0], f32>
        map.yield %2 : f32
      }, (%arg8: !map.index<2>) {
        %2 = map.load %arg6[%arg8] : !mo.tensor<[M, N1], f32>
        map.yield %2 : f32
      }
      map.store %arg7[%arg4] = %1 : !mo.tensor<[M, P], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[M, P], f32>
}

// On an outer axis no vector can cross from one input into the next.

// CHECK-LABEL: mo.graph @concatOuterAxis
// CHECK:       mogg.index.concat.select(%{{.*}}) <axis = 0, sizes = #mosh<ape[2, 3]> : !mosh.ape>
mo.graph @concatOuterAxis(%arg0: !mo.tensor<[2, 8], f32>, %arg1: !mo.tensor<[3, 8], f32>) -> !mo.tensor<[5, 8], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[2, 8], f32>, %arg1: !mo.tensor<[3, 8], f32>)
    outputs (%out : !mo.tensor<[5, 8], f32>) {
    map.iter.elementwise.body<[5, 8]> 
      index   (%arg4: !map.index<2>)
      capturing {%arg5 = %arg0 : !mo.tensor<[2, 8], f32>, %arg6 = %arg1 : !mo.tensor<[3, 8], f32>, %arg7 = %out : !mo.tensor<[5, 8], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[2, 8], [3, 8] -> [5, 8], axis = 0>> (%arg4) : !map.index<2> -> f32 (%arg8: !map.index<2>) {
        %2 = map.load %arg5[%arg8] : !mo.tensor<[2, 8], f32>
        map.yield %2 : f32
      }, (%arg8: !map.index<2>) {
        %2 = map.load %arg6[%arg8] : !mo.tensor<[3, 8], f32>
        map.yield %2 : f32
      }
      map.store %arg7[%arg4] = %1 : !mo.tensor<[5, 8], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[5, 8], f32>
}
