// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// MAP-lane counterpart of codegen-inline-broadcast-constant-scalars.mlir: the
// broadcast constant is emitted as a literal operand of the elementwise call,
// splatted to the closure's SIMD width so it works in any operand position,
// and the kernel takes one input pointer rather than two -- the constant
// buffer is never loaded from, or even passed in.

// CHECK: def {{__[A-Za-z0-9_]+}}([[IN:[^:]*]]: OpaquePointer[MutAnyOrigin], [[OUT:[^:]*]]: OpaquePointer[MutAnyOrigin], [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]([[IN]]{{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]([[OUT]]{{.*}})
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
// CHECK:   def [[LAMBDA:lambda_____[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[XV:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = [[SPEC0]], simd_width = width, element_alignment = element_alignment]([[PTR0]], [[SH0]], [[ST0]], {{.*}})
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], SIMD[DType.float32, width](2.0))
// CHECK:     {{.*}}simd_store_into_tensor_pointer[static_spec = [[SPEC1]], simd_width = width, element_alignment = element_alignment]([[PTR1]], [[SH1]], [[ST1]], {{.*}}, rebind[SIMD[{{.*}}, width]]([[RES]]))
// The constant is inlined, so nothing reads a zero-strided view of it.
// CHECK-NOT: mogg_tensor_create_broadcast

mo.graph @add_scalar_constant(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %s = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[2, 2], f32>
  mo.output %r : !mo.tensor<[2, 2], f32>
}
