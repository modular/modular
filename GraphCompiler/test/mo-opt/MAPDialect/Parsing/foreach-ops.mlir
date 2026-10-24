// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip the map.iter.elementwise family of ops.
// Covers both fully-static shapes and MOSH symbolic-dim shapes.
// Value names are matched with capture variables so changes in the printer's
// SSA numbering don't break the test.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

// map.call.elementwise verifies its callee resolves to a mogg.kernel.decl
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @builtin_kernels.gather_scatter.MutableStore registration = "mo.mutable.store" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// Static shape: verify map.iter.elementwise<[8, 16, 128]> round-trips unchanged.
// CHECK-LABEL: mo.graph @elementwiseAdd
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.elementwise<[8, 16, 128]>
// CHECK-NEXT:    inputs  ([[X:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>, [[Y:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    outputs ([[OUTARG:%[^ :,]+]]: !mo.tensor<[8, 16, 128], f32>)
// CHECK-NEXT:    index   ([[IDX:%[^ :,]+]]: !map.index<3>) -> !mo.tensor<[8, 16, 128], f32> {
// CHECK-NEXT:   [[XV:%[^ ]+]] = map.load [[X]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:   [[YV:%[^ ]+]] = map.load [[Y]][[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT:   [[RES:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX]], [[XV]], [[YV]]) : !map.index<3>, f32, f32 -> f32
// CHECK-NEXT:   map.store [[OUTARG]][[[IDX]]] = [[RES]] : !mo.tensor<[8, 16, 128], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[8, 16, 128], f32>
mo.graph @elementwiseAdd(%x: !mo.tensor<[8, 16, 128], f32>,
                         %y: !mo.tensor<[8, 16, 128], f32>)
    -> !mo.tensor<[8, 16, 128], f32> {
  %output = map.iter.elementwise<[8, 16, 128]>
      inputs  (%x: !mo.tensor<[8, 16, 128], f32>,
               %y: !mo.tensor<[8, 16, 128], f32>)
      outputs (%out: !mo.tensor<[8, 16, 128], f32>)
      index   (%idx: !map.index<3>)
      -> !mo.tensor<[8, 16, 128], f32>
  {
    %xv = map.load %x[%idx] : !mo.tensor<[8, 16, 128], f32>
    %yv = map.load %y[%idx] : !mo.tensor<[8, 16, 128], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[8, 16, 128], f32>
  }
  mo.output %output : !mo.tensor<[8, 16, 128], f32>
}

// Symbolic shape: named MOSH dims in the shape attr.
// CHECK-LABEL: mo.graph @elementwiseAddDynamic
// CHECK:         [[RESULT2:%[^ ]+]] = map.iter.elementwise<[D0, D1]>
// CHECK-NEXT:    inputs  ([[X2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>, [[Y2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    outputs ([[OUTARG2:%[^ :,]+]]: !mo.tensor<[D0, D1], f32>)
// CHECK-NEXT:    index   ([[IDX2:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[D0, D1], f32> {
mo.graph @elementwiseAddDynamic<D0, D1>(
    %x: !mo.tensor<[D0, D1], f32>,
    %y: !mo.tensor<[D0, D1], f32>)
    -> !mo.tensor<[D0, D1], f32> {
  %output = map.iter.elementwise<[D0, D1]>
      inputs  (%x: !mo.tensor<[D0, D1], f32>,
               %y: !mo.tensor<[D0, D1], f32>)
      outputs (%out: !mo.tensor<[D0, D1], f32>)
      index   (%idx: !map.index<2>)
      -> !mo.tensor<[D0, D1], f32>
  {
    %xv = map.load %x[%idx] : !mo.tensor<[D0, D1], f32>
    %yv = map.load %y[%idx] : !mo.tensor<[D0, D1], f32>
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%idx] = %outv : !mo.tensor<[D0, D1], f32>
  }
  mo.output %output : !mo.tensor<[D0, D1], f32>
}

// CHECK-LABEL: mo.graph @mutableStore
// CHECK:         [[OUTCHAIN:%[^ ]+]] = map.iter.elementwise<[2, 2]>
// CHECK-NEXT:    inputs  ([[VAL:%[^ :,]+]]: !mo.tensor<[2, 2], f32>, [[BUF:%[^ :,]+]]: !mo.buffer<[2, 2], f32>)
// CHECK-NEXT:    outputs ()
// CHECK-NEXT:    index   ([[IDX3:%[^ :,]+]]: !map.index<2>) chain([[INCHAIN:%[^)]+]]) -> !mo.chain {
// CHECK-NEXT:   [[VALV:%[^ ]+]] = map.load [[VAL]][[[IDX3]]] : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:   [[RES3:%[^ ]+]] = map.call.elementwise @builtin_kernels.gather_scatter.MutableStore([[IDX3]], [[VALV]]) : !map.index<2>, f32 -> f32
// CHECK-NEXT:   map.store [[BUF]][[[IDX3]]] = [[RES3]] : !mo.buffer<[2, 2], f32>
// CHECK-NEXT: } {hasWriteEffect = true}
// CHECK-NEXT:   mo.output
mo.graph @mutableStore(%val: !mo.tensor<[2, 2], f32>, %buf: !mo.buffer<[2, 2], f32>) {
  %c = mo.chain.create()
  %oc = map.iter.elementwise<[2, 2]>
      inputs  (%val: !mo.tensor<[2, 2], f32>, %buf: !mo.buffer<[2, 2], f32>)
      outputs ()
      index   (%idx: !map.index<2>)
      chain   (%c)
      -> !mo.chain
  {
    %valv = map.load %val[%idx] : !mo.tensor<[2, 2], f32>
    %res = map.call.elementwise @builtin_kernels.gather_scatter.MutableStore(%idx, %valv)
                : !map.index<2>, f32 -> f32
    map.store %buf[%idx] = %res : !mo.buffer<[2, 2], f32>
  } {hasWriteEffect = true}
  mo.output
}

}
