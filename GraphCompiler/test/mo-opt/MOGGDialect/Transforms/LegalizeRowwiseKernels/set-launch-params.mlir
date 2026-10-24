// RUN: mo-opt %s --mogg-rowwise-set-launch-params | FileCheck %s
// RUN: mo-opt %s --mogg-rowwise-set-launch-params="expensive-cost-threshold=32" | FileCheck %s --check-prefix=T32

// Sets each rowwise kernel's launch parameters. Bodies are reduce-shaped, so
// `num_phases` is 1 and `dtype_size` 0. `cols` is the static size of the
// reduced axis.


!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "mo.reduce.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Div registration = "mo.div" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !lit.struct<@std::@builtin::@dtype::@DType>, "width": !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>("lhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>, "rhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>) -> !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>>>, [#kgen<param.decl dtype : !lit.struct<@std::@builtin::@dtype::@DType>>, #kgen<param.decl width : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}, {dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !lit.struct<@std::@builtin::@dtype::@DType>, "width": !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>("lhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>, "rhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>) -> !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>>>, [#kgen<param.decl dtype : !lit.struct<@std::@builtin::@dtype::@DType>>, #kgen<param.decl width : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}, {dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Exp registration = "mo.exp" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !lit.struct<@std::@builtin::@dtype::@DType>, "width": !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>("x": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>) -> !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>>>, [#kgen<param.decl dtype : !lit.struct<@std::@builtin::@dtype::@DType>>, #kgen<param.decl width : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: func.func @staticSum
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {cols = 128 : i64, dtype_size = 0 : i64, num_phases = 1 : i64, simd_monoid = @builtin_kernels.monoids.ReduceSum}
func.func @staticSum(%ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// A dynamic axis has no `cols`.
// CHECK-LABEL: func.func @dynamicArgMax
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {dtype_size = 0 : i64, num_phases = 1 : i64, simd_monoid = @builtin_kernels.monoids.ArgMax}
func.func @dynamicArgMax(%ctx: !mogg.context, %m: !mogg.dim, %n: !mogg.dim, %x: !mogg.tensor, %ov: !mogg.tensor, %oi: !mogg.tensor) {
  %shape = mogg.shape.create(%m, %n) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape> : !mogg.dim, !mogg.dim
  mogg.call.rowwise shape(%shape) axis(0) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cv = %ov : !mogg.tensor, %ci = %oi : !mogg.tensor} device = "cpu" {
    %v, %i = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (!mogg.simd<f32>, !mogg.simd<si64>) {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %cv[%row] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      mogg.tensor.store %ci[%row] = %i : !mogg.tensor, !map.index<2>, !mogg.simd<si64>
    }
  }
  return
}

// The SIMD width comes from the fold with the largest state: ArgMax's f32 and
// si64 outweigh ReduceSum's f32.
// CHECK-LABEL: func.func @widestState
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  simd_monoid = @builtin_kernels.monoids.ArgMax
func.func @widestState(%ctx: !mogg.context, %x: !mogg.tensor, %os: !mogg.tensor, %oi: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cs = %os : !mogg.tensor, %ci = %oi : !mogg.tensor} device = "cpu" {
    %s = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    %v, %i = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (!mogg.simd<f32>, !mogg.simd<si64>) {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %xv : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %cs[%row] = %s : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      mogg.tensor.store %ci[%row] = %i : !mogg.tensor, !map.index<2>, !mogg.simd<si64>
    }
  }
  return
}

// An exp in the fold runs per element and costs 16: the kernel is
// computationally expensive, unless the threshold is higher.
// CHECK-LABEL: func.func @expensiveFold
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {cols = 128 : i64, computationally_expensive, dtype_size
// T32-LABEL:   func.func @expensiveFold
// T32:         mogg.call.rowwise
// T32-SAME:    attributes {cols = 128 : i64, dtype_size
func.func @expensiveFold(%ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %e = mogg.call.elementwise @builtin_kernels.elementwise.Exp(%v) moOpName = "mo.exp" : !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.yield %e : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// A division after the fold runs once per row: it doesn't make the kernel
// expensive.
// CHECK-LABEL: func.func @expensiveRowOp
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {cols = 128 : i64, dtype_size
func.func @expensiveRowOp(%ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    %n = mogg.constant.scalar 1.280000e+02 : f32 : !mogg.simd<f32>
    %m = mogg.call.elementwise @builtin_kernels.elementwise.Div(%acc, %n) moOpName = "mo.div" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
    mogg.rowwise.emit {
      mogg.tensor.store %co[%row] = %m : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// Cheap ops cost 1 each: 15 of them stay under the threshold.
// CHECK-LABEL: func.func @fifteenCheapOps
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {cols = 128 : i64, dtype_size
func.func @fifteenCheapOps(%ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %a0 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%v, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a1 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a0, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a2 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a1, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a3 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a2, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a4 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a3, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a5 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a4, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a6 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a5, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a7 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a6, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a8 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a7, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a9 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a8, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a10 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a9, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a11 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a10, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a12 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a11, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a13 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a12, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a14 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a13, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.yield %a14 : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// 16 cheap ops reach it.
// CHECK-LABEL: func.func @sixteenCheapOps
// CHECK:       mogg.call.rowwise
// CHECK-SAME:  attributes {cols = 128 : i64, computationally_expensive, dtype_size
func.func @sixteenCheapOps(%ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  %shape = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %a0 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%v, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a1 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a0, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a2 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a1, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a3 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a2, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a4 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a3, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a5 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a4, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a6 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a5, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a7 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a6, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a8 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a7, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a9 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a8, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a10 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a9, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a11 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a10, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a12 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a11, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a13 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a12, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a14 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a13, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      %a15 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%a14, %v) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.yield %a15 : !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}
