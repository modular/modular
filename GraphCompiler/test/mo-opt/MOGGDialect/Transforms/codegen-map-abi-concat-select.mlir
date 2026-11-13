// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages \
// RUN:   --map-infer-max-simd-width \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// A concat's foreach closure emits its select as one nested closure over the
// input index, a `comptime if` branch per region, each reading its input as
// wide as the foreach's own vectors, and hands it to the primitive. Since a
// vector must not cross from one input into the next, the width must divide
// every input's extent: `_foreach` gets their GCD, as an `Idx` when the
// extents are static, from the inputs' shapes at run time when they
// are not.

// CHECK:      def lambda_{{_+[0-9]+}}[width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK-NEXT:   var [[IDX:var_+[0-9]+]] = std.utils.coord.coord_to_index_list(idx)
// CHECK-NEXT:   @inline(.always)
// CHECK-NEXT:   def [[INPUTS:concat_inputs_+[0-9]+]][input_index : Int](i: layout.Coord) {var} -> SIMD[DType.float32, width]:
// CHECK-NEXT:     comptime if input_index == 0:
// CHECK-NEXT:       var [[X0:var_+[0-9]+]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = width, element_alignment = element_alignment]({{.*}}std.utils.coord.coord_to_index_list(i)))
// CHECK-NEXT:       return rebind[SIMD[DType.float32, width]]([[X0]])
// CHECK:          else:
// CHECK-NEXT:       var [[X1:var_+[0-9]+]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = width, element_alignment = element_alignment]({{.*}}std.utils.coord.coord_to_index_list(i)))
// CHECK-NEXT:       return rebind[SIMD[DType.float32, width]]([[X1]])
// CHECK:        comptime [[SIZES:param_+[0-9]+]] = layout.Coord(layout.Idx[8], layout.Idx[16])
// CHECK-NEXT:   var [[V:var_+[0-9]+]] = builtin_primitives.primitives.mogg_index_concat_select[dtype = DType.float32, width = width, num_inputs = 2, axis = 1, static_sizes = [[SIZES]]]([[IDX]], [[INPUTS]])
// CHECK-NEXT:   managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}]({{.*}}, rebind[SIMD[{{.*}}, width]]([[V]]))
// CHECK:      builtin_primitives.primitives._foreach[simd_width = builtin_primitives.primitives.get_simd_width_for_dtypes[{{.*}}]()]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, max_simd_width = layout.Idx[8])
mo.graph @concatStatic(%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %0 = map.iter.opaque
    inputs (%arg0: !mo.tensor<[4, 8], f32>, %arg1: !mo.tensor<[4, 16], f32>)
    outputs (%out : !mo.tensor<[4, 24], f32>) {
    map.iter.elementwise.body<[4, 24]> 
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

// CHECK:      def __{{[a-z_0-9]+}}(arg__0: {{[^,]*}}, arg__1: {{[^,]*}}, arg__2: {{[^,]*}}, [[M:arg__[0-9]+]]: Int, [[N0:arg__[0-9]+]]: Int, [[N1:arg__[0-9]+]]: Int,
// CHECK:      var [[S0:var__[0-9]+]] = layout.Coord(Scalar[DType.int]([[M]]), Scalar[DType.int]([[N0]]))
// CHECK:      var [[S1:var__[0-9]+]] = layout.Coord(Scalar[DType.int]([[M]]), Scalar[DType.int]([[N1]]))
// CHECK:      var [[RT:var_+[0-9]+]] = std.utils.IndexList[2](std.utils.coord.coord_to_index_list([[S0]])[1], std.utils.coord.coord_to_index_list([[S1]])[1])
// CHECK-NEXT: comptime [[DSIZES:param_+[0-9]+]] = layout.Coord(Int(-1), Int(-1))
// CHECK-NEXT: var {{var_+[0-9]+}} = builtin_primitives.primitives.mogg_index_concat_select[dtype = DType.float32, width = width, num_inputs = 2, axis = 1, static_sizes = [[DSIZES]]]({{var_+[0-9]+}}, {{concat_inputs_+[0-9]+}}, [[RT]])
// CHECK:      var [[G:var_+[0-9]+]] = builtin_primitives.primitives.pow2_divisor_of_dims(std.utils.coord.coord_to_index_list({{var_+[0-9]+}}))
// CHECK-NEXT: builtin_primitives.primitives._foreach[simd_width = {{.*}}]({{.*}}, max_simd_width = Scalar[DType.int]([[G]]))
mo.graph @concatDynamic<M, N0, N1, P>(%arg0: !mo.tensor<[M, N0], f32>, %arg1: !mo.tensor<[M, N1], f32>) -> !mo.tensor<[M, P], f32> {
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
