// RUN: mo-opt %s --mogg-rowwise-group-row-stores | FileCheck %s

// The stores at the row index move into one `mogg.rowwise.emit` at the end of
// the body.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "mo.reduce.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "mo.reduce.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: func.func @argMax
// CHECK:       %[[F:.*]]:2 = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax
// CHECK:       }
// CHECK-NEXT:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[F]]#0 : !mogg.simd<f32> to !mogg.simd<f32>
// CHECK-NEXT:  mogg.rowwise.emit {
// CHECK-NEXT:    mogg.tensor.store %{{.*}}[%row] = %[[F]]#1
// CHECK-NEXT:    mogg.tensor.store %{{.*}}[%row] = %[[N]]
// CHECK-NEXT:  }
// CHECK-NEXT: }
func.func @argMax(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %oi: !mogg.tensor, %on: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %ci = %oi : !mogg.tensor, %cn = %on : !mogg.tensor} device = "cpu" {
    %v, %i = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (!mogg.simd<f32>, !mogg.simd<si64>) {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    mogg.tensor.store %ci[%row] = %i : !mogg.tensor, !map.index<2>, !mogg.simd<si64>
    %n = builtin.unrealized_conversion_cast %v : !mogg.simd<f32> to !mogg.simd<f32>
    mogg.tensor.store %cn[%row] = %n : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}
