// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[DIM0:.*]]: Int, [[DIM1:.*]]: Int, {{.*}}: Int, [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[SHAPE:.*]] = layout.Coord(Scalar[DType.int]([[DIM0]]), Scalar[DType.int]([[DIM1]]))
// CHECK:   var {{.*}} = layout.row_major([[SHAPE]])
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}, {{.*}})
// CHECK:   var [[MTS2:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]({{.*}}, {{.*}})
// `map-flatten-shapes` merges the two symbolic dims, so each tensor is read
// through a collapse view and the loop runs at rank 1.
// CHECK:   var [[COL0:.*]] = {{.*}}mogg_tensor_create_collapse[{{.*}}]({{.*}})
// CHECK:   var [[COL1:.*]] = {{.*}}mogg_tensor_create_collapse[{{.*}}]({{.*}})
// CHECK:   var [[COL2:.*]] = {{.*}}mogg_tensor_create_collapse[{{.*}}]({{.*}})
// CHECK:   var [[FLAT:.*]] = layout.Coord(Scalar[DType.int]({{.*}}))
// Each capture is decomposed into (ptr, shape, strides) plus its comptime
// StaticTensorSpec, and accessed through the pointer-based simd load/store -- no
// TileTensor is materialized.
// CHECK:   var [[PTR0:.*]] = [[COL0]].unsafe_ptr()
// CHECK:   var [[SH0:.*]] = [[COL0]].shape_coord()
// CHECK:   var [[ST0:.*]] = [[COL0]].strides_coord()
// CHECK:   comptime [[SPEC0:.*]] = [[COL0]].static_spec
// CHECK:   var [[PTR1:.*]] = [[COL1]].unsafe_ptr()
// CHECK:   var [[SH1:.*]] = [[COL1]].shape_coord()
// CHECK:   var [[ST1:.*]] = [[COL1]].strides_coord()
// CHECK:   comptime [[SPEC1:.*]] = [[COL1]].static_spec
// CHECK:   var [[PTR2:.*]] = [[COL2]].unsafe_ptr()
// CHECK:   var [[SH2:.*]] = [[COL2]].shape_coord()
// CHECK:   var [[ST2:.*]] = [[COL2]].strides_coord()
// CHECK:   comptime [[SPEC2:.*]] = [[COL2]].static_spec
// CHECK:   def [[LAMBDA:lambda_____[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[XV:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = [[SPEC0]], simd_width = width, element_alignment = element_alignment]([[PTR0]], [[SH0]], [[ST0]], {{.*}})
// CHECK:     var [[YV:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = [[SPEC1]], simd_width = width, element_alignment = element_alignment]([[PTR1]], [[SH1]], [[ST1]], {{.*}})
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], [[YV]])
// CHECK:     {{.*}}simd_store_into_tensor_pointer[static_spec = [[SPEC2]], simd_width = width, element_alignment = element_alignment]([[PTR2]], [[SH2]], [[ST2]], {{.*}}, rebind[SIMD[{{.*}}, width]]([[RES]]))
// CHECK:   {{.*}}_foreach[simd_width = {{.*}}get_simd_width_for_dtypes[{{.*}}]()]([[FLAT]], [[LAMBDA]], [[CTX]])

mo.graph @add_dynamic<M, N>(%a: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[M, N], f32>) {
  %c = mo.add(%a, %b) : !mo.tensor<[M, N], f32>
  mo.output %c : !mo.tensor<[M, N], f32>
}
