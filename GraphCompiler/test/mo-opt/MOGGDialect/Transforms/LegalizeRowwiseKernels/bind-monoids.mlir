// RUN: mo-opt %s --mogg-rowwise-bind-monoids | FileCheck %s

// Each fold's monoid dtype parameters are bound from its result types, and
// its output fields are named. `accum` is the widest contribution type.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "mo.reduce.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "mo.reduce.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK-LABEL: func.func @bf16Sum
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc"], params = {dtype = f32}}
func.func @bf16Sum(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<bf16>
      %a = builtin.unrealized_conversion_cast %v : !mogg.simd<bf16> to !mogg.simd<f32>
      mogg.yield %a : !mogg.simd<f32>
    }
    mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// ArgMax's index field has a fixed int64 dtype; only `dtype` is bound.
// CHECK-LABEL: func.func @argMax
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc_values", "acc_indices"], params = {dtype = f32}}
func.func @argMax(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %ov: !mogg.tensor, %oi: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cv = %ov : !mogg.tensor, %ci = %oi : !mogg.tensor} device = "cpu" {
    %v, %i = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (!mogg.simd<f32>, !mogg.simd<si64>) {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    mogg.tensor.store %cv[%row] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    mogg.tensor.store %ci[%row] = %i : !mogg.tensor, !map.index<2>, !mogg.simd<si64>
  }
  return
}

// `accum` is the wider of the two folds' contributions.
// CHECK-LABEL: func.func @twoFolds
// CHECK:       attributes {accum = f32}
// CHECK:       } {fields = ["acc"], params = {dtype = f16}}
// CHECK:       } {fields = ["acc"], params = {dtype = f32}}
func.func @twoFolds(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %a = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f16> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f16>
      mogg.yield %v : !mogg.simd<f16>
    }
    %b = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f16>
      %w = builtin.unrealized_conversion_cast %v : !mogg.simd<f16> to !mogg.simd<f32>
      mogg.yield %w : !mogg.simd<f32>
    }
    %af = builtin.unrealized_conversion_cast %a : !mogg.simd<f16> to !mogg.simd<f32>
    %s = builtin.unrealized_conversion_cast %af, %b : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
    mogg.tensor.store %co[%row] = %s : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}
