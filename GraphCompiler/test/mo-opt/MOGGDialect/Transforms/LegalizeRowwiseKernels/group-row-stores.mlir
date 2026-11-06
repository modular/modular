// RUN: mo-opt %s --mogg-rowwise-group-row-stores | FileCheck %s

// The stores at the row index, and the row-level ops only they read, move into
// one `mogg.rowwise.emit` at the end of the body. A load at the row index that
// a phase reads stays in the body: a `mogg.rowwise.load_lane` spread by a
// `mogg.rowwise.splat_lane`.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "reduction.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "reduction.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: func.func @argMax
// CHECK:       %[[F:.*]]:2 = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax
// CHECK:       }
// CHECK-NEXT:  mogg.rowwise.emit {
// CHECK-NEXT:    %[[N:.*]] = builtin.unrealized_conversion_cast %[[F]]#0 : !mogg.simd<f32> to !mogg.simd<f32>
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

// CHECK-LABEL: func.func @rowLoadReadByPhase
// CHECK:       %[[ACC:.*]] = mogg.rowwise.reduce
// CHECK:       }
// CHECK-NEXT:  %[[L:.*]] = mogg.rowwise.load_lane %{{.*}}[%row] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
// CHECK-NEXT:  %[[B:.*]] = mogg.rowwise.splat_lane %[[L]] : !mogg.simd<f32>
// CHECK-NEXT:  %[[S:.*]] = builtin.unrealized_conversion_cast %[[ACC]], %[[B]]
// CHECK-NEXT:  mogg.rowwise.map
// CHECK-NOT:   mogg.rowwise.emit
func.func @rowLoadReadByPhase(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %b: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cb = %b : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    %bv = mogg.tensor.load %cb[%row] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
    %s = builtin.unrealized_conversion_cast %acc, %bv : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
    mogg.rowwise.map (%j : !map.index<2>) {
      %xv = mogg.tensor.load %cx[%j] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %y = builtin.unrealized_conversion_cast %xv, %s : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %y : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}
