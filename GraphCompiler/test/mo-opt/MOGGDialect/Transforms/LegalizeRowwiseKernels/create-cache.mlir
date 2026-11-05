// RUN: mo-opt %s --mogg-rowwise-create-cache | FileCheck %s

// Without a `mogg.rowwise.cache` from MAP, the row caches the capture the first
// fold loads at its own index, and every phase's load of it there becomes
// `mogg.rowwise.cache.load`; other loads stay. `is_cached` says whether more
// than one phase reads the cache.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "reduction.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "reduction.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Div registration = "mo.div" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !lit.struct<@std::@builtin::@dtype::@DType>, "width": !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>("lhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>, "rhs": !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>) -> !lit.struct<@std::@simd::@SIMD<:!lit.struct<@std::@builtin::@dtype::@DType> *(0,0), :!lit.struct<@std::@builtin::@simd_length::@SIMDLength> *(0,1)>>>>, [#kgen<param.decl dtype : !lit.struct<@std::@builtin::@dtype::@DType>>, #kgen<param.decl width : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>>], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}, {dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}], [{dtype = #kgen.param.decl.ref<"dtype"> : !lit.struct<@std::@builtin::@dtype::@DType>, size = #kgen.param.decl.ref<"width"> : !lit.struct<@std::@builtin::@simd_length::@SIMDLength>}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: func.func @twoInputs
// CHECK:       mogg.call.rowwise {{.*}} capturing {%[[X:cap[0-9]+]] = %arg{{[0-9]+}} : !mogg.tensor, %[[Y:cap[0-9]+]] = %arg{{[0-9]+}} : !mogg.tensor, {{.*}}} device = "cpu" {
// CHECK-NEXT:    mogg.rowwise.cache (%[[K:[[:alnum:]_]+]] : !map.index<2>) -> !mogg.simd<f32> {
// CHECK-NEXT:      %[[XK:.*]] = mogg.tensor.load %[[X]][%[[K]]]
// CHECK-NEXT:      mogg.yield %[[XK]] : !mogg.simd<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%[[I:[[:alnum:]_]+]] : !map.index<2>) -> !mogg.simd<f32> {
// CHECK-NEXT:      %[[XV:.*]] = mogg.rowwise.cache.load : !mogg.simd<f32>
// CHECK-NEXT:      %[[YV:.*]] = mogg.tensor.load %[[Y]][%[[I]]]
// CHECK-NEXT:      builtin.unrealized_conversion_cast %[[XV]], %[[YV]]
// CHECK:         mogg.tensor.load %[[X]][%row]
func.func @twoInputs(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %yv = mogg.tensor.load %cy[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %p = builtin.unrealized_conversion_cast %xv, %yv : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.yield %p : !mogg.simd<f32>
    }
    %xr = mogg.tensor.load %cx[%row] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
    %s = builtin.unrealized_conversion_cast %acc, %xr : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
    mogg.tensor.store %co[%row] = %s : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// The first load picks the input, whatever the capture order.
// CHECK-LABEL: func.func @firstLoad
// CHECK:       mogg.call.rowwise {{.*}} capturing {%{{.*}}, %[[Y:cap[0-9]+]] = %arg{{[0-9]+}} : !mogg.tensor, {{.*}}} device = "cpu" {
// CHECK-NEXT:    mogg.rowwise.cache
// CHECK-NEXT:      mogg.tensor.load %[[Y]]
func.func @firstLoad(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %yv = mogg.tensor.load %cy[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %xv = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %p = builtin.unrealized_conversion_cast %xv, %yv : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.yield %p : !mogg.simd<f32>
    }
    mogg.tensor.store %co[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// Every fold's element loads of the input are replaced.
// CHECK-LABEL: func.func @twoFolds
// CHECK:       attributes {is_cached}
// CHECK:       mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:    mogg.rowwise.cache.load : !mogg.simd<f32>
// CHECK:       mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:    mogg.rowwise.cache.load : !mogg.simd<f32>
func.func @twoFolds(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %a = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    %b = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %d = builtin.unrealized_conversion_cast %v, %a : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.yield %d : !mogg.simd<f32>
    }
    mogg.tensor.store %co[%row] = %b : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// A map phase reads the input too, so the row is cached.
// CHECK-LABEL: func.func @mapPhase
// CHECK:       attributes {is_cached}
// CHECK:       mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
// CHECK-NEXT:    mogg.rowwise.cache.load : !mogg.simd<f32>
// CHECK:       mogg.rowwise.map (%[[J:[[:alnum:]_]+]] : !map.index<2>) {
// CHECK-NEXT:    %[[XV:.*]] = mogg.rowwise.cache.load : !mogg.simd<f32>
// CHECK-NEXT:    %[[Y:.*]] = builtin.unrealized_conversion_cast %[[XV]], %{{.*}}
// CHECK-NEXT:    mogg.tensor.store %{{.*}}[%[[J]]] = %[[Y]]
func.func @mapPhase(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.rowwise.map (%j : !map.index<2>) {
      %v = mogg.tensor.load %cx[%j] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %y = builtin.unrealized_conversion_cast %v, %acc : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %y : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// A cache from MAP stays; the phases that read it set `is_cached`.
// CHECK-LABEL: func.func @cacheFromMap
// CHECK:       attributes {is_cached}
// CHECK-NEXT:    mogg.rowwise.cache
// CHECK-NEXT:      mogg.tensor.load
// CHECK-NEXT:      mogg.tensor.load
// CHECK-NEXT:      mogg.call.elementwise @builtin_kernels.elementwise.Div
// CHECK-NOT:     mogg.rowwise.cache (
func.func @cacheFromMap(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %p = mogg.rowwise.cache (%k : !map.index<2>) -> !mogg.simd<f32> {
      %xv = mogg.tensor.load %cx[%k] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %yv = mogg.tensor.load %cy[%k] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %q = mogg.call.elementwise @builtin_kernels.elementwise.Div(%xv, %yv) moOpName = "mo.div" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.yield %q : !mogg.simd<f32>
    }
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.rowwise.cache.load : !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.rowwise.map (%j : !map.index<2>) {
      %v = mogg.rowwise.cache.load : !mogg.simd<f32>
      %r = builtin.unrealized_conversion_cast %v, %acc : !mogg.simd<f32>, !mogg.simd<f32> to !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %r : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}
