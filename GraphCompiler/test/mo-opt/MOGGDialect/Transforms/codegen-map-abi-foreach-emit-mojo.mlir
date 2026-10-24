// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s


// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[SHAPE:.*]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:   var {{.*}} = layout.row_major([[SHAPE]])
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS2:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]({{.*}}, {{.*}})
// Each capture is decomposed into (ptr, shape, strides) plus its comptime
// StaticTensorSpec, and accessed through the pointer-based simd load/store -- no
// TileTensor is materialized.
// CHECK:   var [[PTR0:.*]] = [[MTS0]].unsafe_ptr()
// CHECK:   var [[SH0:.*]] = [[MTS0]].shape()
// CHECK:   var [[ST0:.*]] = [[MTS0]].strides()
// CHECK:   comptime [[SPEC0:.*]] = [[MTS0]].static_spec
// CHECK:   var [[PTR1:.*]] = [[MTS1]].unsafe_ptr()
// CHECK:   var [[SH1:.*]] = [[MTS1]].shape()
// CHECK:   var [[ST1:.*]] = [[MTS1]].strides()
// CHECK:   comptime [[SPEC1:.*]] = [[MTS1]].static_spec
// CHECK:   var [[PTR2:.*]] = [[MTS2]].unsafe_ptr()
// CHECK:   var [[SH2:.*]] = [[MTS2]].shape()
// CHECK:   var [[ST2:.*]] = [[MTS2]].strides()
// CHECK:   comptime [[SPEC2:.*]] = [[MTS2]].static_spec
// CHECK:   def [[LAMBDA:lambda_____[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[XV:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = [[SPEC0]], simd_width = width, element_alignment = element_alignment]([[PTR0]], [[SH0]], [[ST0]], {{.*}})
// CHECK:     var [[YV:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = [[SPEC1]], simd_width = width, element_alignment = element_alignment]([[PTR1]], [[SH1]], [[ST1]], {{.*}})
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], [[YV]])
// CHECK:     {{.*}}simd_store_into_tensor_pointer[static_spec = [[SPEC2]], simd_width = width, element_alignment = element_alignment]([[PTR2]], [[SH2]], [[ST2]], {{.*}}, rebind[SIMD[{{.*}}, width]]([[RES]]))
// CHECK:   {{.*}}_foreach[simd_width = {{.*}}get_simd_width_for_dtypes[{{.*}}](), target = "cpu"]([[SHAPE]], [[LAMBDA]], [[CTX]])

mo.graph @add(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.add(%arg0, %arg1) : !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}
