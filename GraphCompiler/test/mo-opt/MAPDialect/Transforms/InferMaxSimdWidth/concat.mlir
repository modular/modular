// RUN: mo-opt %s --map-infer-max-simd-width | FileCheck %s

// A concat select along the innermost axis reads each vector from one input,
// so the width must divide every input's extent: the loop's max SIMD width is
// their GCD, folded when static, symbolic when not. Along an outer axis a
// vector stays in one input. A nested concat's limit holds as it is.

// CHECK-LABEL: mo.graph @innermost_static
// CHECK:       map.iter.elementwise.body<[4, 16]> max_simd_width(2)
mo.graph @innermost_static(%arg0: !mo.tensor<[4, 6], f32>, %arg1: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[4, 6], f32>, %arg1: !mo.tensor<[4, 10], f32>)
    outputs (%out : !mo.tensor<[4, 16], f32>) {
    map.iter.elementwise.body<[4, 16]> 
      index   (%arg4: !map.index<2>)
      capturing {%arg5 = %arg0 : !mo.tensor<[4, 6], f32>, %arg6 = %arg1 : !mo.tensor<[4, 10], f32>, %arg7 = %out : !mo.tensor<[4, 16], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[4, 6], [4, 10] -> [4, 16], axis = 1>> (%arg4) : !map.index<2> -> f32 (%arg8: !map.index<2>) {
        %2 = map.load %arg5[%arg8] : !mo.tensor<[4, 6], f32>
        map.yield %2 : f32
      }, (%arg8: !map.index<2>) {
        %2 = map.load %arg6[%arg8] : !mo.tensor<[4, 10], f32>
        map.yield %2 : f32
      }
      map.store %arg7[%arg4] = %1 : !mo.tensor<[4, 16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[4, 16], f32>
}

// CHECK-LABEL: mo.graph @innermost_dynamic
// CHECK:       map.iter.elementwise.body<[M, P]> max_simd_width(#map.gcd<[N0, N1]>)
mo.graph @innermost_dynamic<M, N0, N1, P>(%arg0: !mo.tensor<[M, N0], f32>, %arg1: !mo.tensor<[M, N1], f32>) -> !mo.tensor<[M, P], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[M, N0], f32>, %arg1: !mo.tensor<[M, N1], f32>)
    outputs (%out : !mo.tensor<[M, P], f32>) {
    map.iter.elementwise.body<[M, P]> 
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

// CHECK-LABEL: mo.graph @outer_axis
// CHECK:       map.iter.elementwise.body
// CHECK-NOT:   max_simd_width
// CHECK:       mo.output
mo.graph @outer_axis(%arg0: !mo.tensor<[3, 16], f32>, %arg1: !mo.tensor<[5, 16], f32>) -> !mo.tensor<[8, 16], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[3, 16], f32>, %arg1: !mo.tensor<[5, 16], f32>)
    outputs (%out : !mo.tensor<[8, 16], f32>) {
    map.iter.elementwise.body<[8, 16]> 
      index   (%arg4: !map.index<2>)
      capturing {%arg5 = %arg0 : !mo.tensor<[3, 16], f32>, %arg6 = %arg1 : !mo.tensor<[5, 16], f32>, %arg7 = %out : !mo.tensor<[8, 16], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[3, 16], [5, 16] -> [8, 16], axis = 0>> (%arg4) : !map.index<2> -> f32 (%arg8: !map.index<2>) {
        %2 = map.load %arg5[%arg8] : !mo.tensor<[3, 16], f32>
        map.yield %2 : f32
      }, (%arg8: !map.index<2>) {
        %2 = map.load %arg6[%arg8] : !mo.tensor<[5, 16], f32>
        map.yield %2 : f32
      }
      map.store %arg7[%arg4] = %1 : !mo.tensor<[8, 16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[8, 16], f32>
}

// An outer concat of two innermost concats: the inner ones' extents count.
// CHECK-LABEL: mo.graph @nested
// CHECK:       map.iter.elementwise.body<[4, 16]> max_simd_width(4)
mo.graph @nested(%arg0: !mo.tensor<[2, 4], f32>, %arg1: !mo.tensor<[2, 12], f32>, %arg2: !mo.tensor<[2, 8], f32>, %arg3: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[4, 16], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[2, 4], f32>, %arg1: !mo.tensor<[2, 12], f32>, %arg2: !mo.tensor<[2, 8], f32>, %arg3: !mo.tensor<[2, 8], f32>)
    outputs (%out : !mo.tensor<[4, 16], f32>) {
    map.iter.elementwise.body<[4, 16]> 
      index   (%arg8: !map.index<2>)
      capturing {%arg9 = %arg0 : !mo.tensor<[2, 4], f32>, %arg10 = %arg1 : !mo.tensor<[2, 12], f32>, %arg11 = %arg2 : !mo.tensor<[2, 8], f32>, %arg12 = %arg3 : !mo.tensor<[2, 8], f32>, %arg13 = %out : !mo.tensor<[4, 16], f32>} {
      %1 = map.index.concat.select<#map.index.concat.spec<[2, 16], [2, 16] -> [4, 16], axis = 0>> (%arg8) : !map.index<2> -> f32 (%arg14: !map.index<2>) {
        %2 = map.index.concat.select<#map.index.concat.spec<[2, 4], [2, 12] -> [2, 16], axis = 1>> (%arg14) : !map.index<2> -> f32 (%arg15: !map.index<2>) {
          %3 = map.load %arg9[%arg15] : !mo.tensor<[2, 4], f32>
          map.yield %3 : f32
        }, (%arg15: !map.index<2>) {
          %3 = map.load %arg10[%arg15] : !mo.tensor<[2, 12], f32>
          map.yield %3 : f32
        }
        map.yield %2 : f32
      }, (%arg14: !map.index<2>) {
        %2 = map.index.concat.select<#map.index.concat.spec<[2, 8], [2, 8] -> [2, 16], axis = 1>> (%arg14) : !map.index<2> -> f32 (%arg15: !map.index<2>) {
          %3 = map.load %arg11[%arg15] : !mo.tensor<[2, 8], f32>
          map.yield %3 : f32
        }, (%arg15: !map.index<2>) {
          %3 = map.load %arg12[%arg15] : !mo.tensor<[2, 8], f32>
          map.yield %3 : f32
        }
        map.yield %2 : f32
      }
      map.store %arg13[%arg8] = %1 : !mo.tensor<[4, 16], f32>
    }
  } {device = #M.device_ref<"cpu", 0>, hasReadEffect = false, hasWriteEffect = false}
  mo.output %0 : !mo.tensor<[4, 16], f32>
}
