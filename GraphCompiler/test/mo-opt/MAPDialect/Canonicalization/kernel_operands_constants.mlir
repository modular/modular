// RUN: mo-opt %s --canonicalize | FileCheck %s

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

mogg.kernel.decl @my_matmul registration = "my_matmul" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @my_add registration = "my_add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false


// CHECK-LABEL: mo.graph @folds_constant_into_opaque_call
// CHECK-NOT:     mo.constant
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[X:%[^ :,]+]]: !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:  outputs ([[O:%[^ :,]+]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NEXT:    [[W:%[^ ]+]] = map.constant
// CHECK:         map.call.opaque @my_matmul[] inputs([[X]], [[W]]) outputs([[O]])
mo.graph @folds_constant_into_opaque_call(
  %x: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], f32> {
  %weights = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %y = map.iter.opaque inputs (%x : !mo.tensor<[2, 2], f32>, %weights as %w : !mo.tensor<[2, 2], f32>) outputs (%out : !mo.tensor<[2, 2], f32>) {
    map.call.opaque @my_matmul[] inputs (%x, %w) outputs (%out) kernelName = "my_matmul" : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @folds_constant_into_foreach
// CHECK-NOT:     mo.constant
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:  outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[2, 2], f32>) {
// CHECK-NEXT:    [[B:%[^ ]+]] = map.constant
// The body is a closure, so the rematerialized constant reaches it the same way
// the kernel's own operands do -- as a capture, replacing the one that used to
// bind the dropped input.
// CHECK-NEXT:    map.iter.elementwise.body<[2, 2]>
// CHECK-NEXT:      index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:      capturing {[[ACAP:%[^ ]+]] = [[A]] : !mo.tensor<[2, 2], f32>, [[BCAP:%[^ ]+]] = [[B]] : !mo.tensor<[2, 2], f32>, [[OCAP:%[^ ]+]] = [[OUT]] : !mo.tensor<[2, 2], f32>} {
// CHECK-NEXT:      [[AV:%[^ ]+]] = map.load [[ACAP]][[[IDX]]]
// CHECK-NEXT:      [[BV:%[^ ]+]] = map.load [[BCAP]][[[IDX]]]
// CHECK-NEXT:      [[SUM:%[^ ]+]] = map.call.elementwise @my_add([[IDX]], [[AV]], [[BV]])
// CHECK-NEXT:      map.store [[OCAP]][[[IDX]]] = [[SUM]]
mo.graph @folds_constant_into_foreach(
  %a: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], f32> {
  %bias = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[2, 2], f32>, %bias as %b : !mo.tensor<[2, 2], f32>)
      outputs (%out : !mo.tensor<[2, 2], f32>) {
    map.iter.elementwise.body<[2, 2]>
        index (%idx: !map.index<2>)
        capturing {%a0 = %a : !mo.tensor<[2, 2], f32>,
                   %b0 = %b : !mo.tensor<[2, 2], f32>,
                   %out0 = %out : !mo.tensor<[2, 2], f32>} {
      %av = map.load %a0[%idx] : !mo.tensor<[2, 2], f32>
      %bv = map.load %b0[%idx] : !mo.tensor<[2, 2], f32>
      %sum = map.call.elementwise @my_add(%idx, %av, %bv)
                : !map.index<2>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[2, 2], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[2, 2], f32>
}

// A `mo.rebind` between the constant and the kernel only reinterprets the
// type, so the constant behind it still folds -- matching
// `MOGG::FuseConstants` on the pre-advanced-fusion lane.
// CHECK-LABEL: mo.graph @folds_constant_through_rebind
// CHECK-NOT:     mo.constant
// CHECK-NOT:     mo.rebind
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[D0], f32>)
// CHECK-NEXT:  outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[D0], f32>) {
// The rematerialized constant carries the dropped input's type rather than the
// constant's own `[4]`, so the body's captures and loads stay well typed.
// CHECK-NEXT:    [[B:%[^ ]+]] = map.constant {{.*}} : tensor<4xf32> : <[D0], f32>
// CHECK-NEXT:    map.iter.elementwise.body<[D0]>
// CHECK-NEXT:      index   ([[IDX:%[^ :,]+]]: !map.index<1>)
// CHECK-NEXT:      capturing {[[ACAP:%[^ ]+]] = [[A]] : !mo.tensor<[D0], f32>, [[BCAP:%[^ ]+]] = [[B]] : !mo.tensor<[D0], f32>, [[OCAP:%[^ ]+]] = [[OUT]] : !mo.tensor<[D0], f32>} {
mo.graph @folds_constant_through_rebind<D0>(
  %a: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D0], f32> {
  %bias = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %rb = mo.rebind(%bias) : !mo.tensor<[4], f32> -> !mo.tensor<[D0], f32>
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[D0], f32>, %rb as %b : !mo.tensor<[D0], f32>)
      outputs (%out : !mo.tensor<[D0], f32>) {
    map.iter.elementwise.body<[D0]>
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[D0], f32>,
                   %b0 = %b : !mo.tensor<[D0], f32>,
                   %out0 = %out : !mo.tensor<[D0], f32>} {
      %av = map.load %a0[%idx] : !mo.tensor<[D0], f32>
      %bv = map.load %b0[%idx] : !mo.tensor<[D0], f32>
      %sum = map.call.elementwise @my_add(%idx, %av, %bv)
                : !map.index<1>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[D0], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[D0], f32>
}

// The look-through reaches a constant only: a rebind of anything else leaves
// the operand on the kernel.
// CHECK-LABEL: mo.graph @keeps_non_constant_through_rebind
// CHECK:         [[RB:%[^ ]+]] = mo.rebind
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[D0], f32>, [[RB]]: !mo.tensor<[D0], f32>)
// CHECK-NOT:     map.constant
mo.graph @keeps_non_constant_through_rebind<D0>(
  %a: !mo.tensor<[D0], f32>,
  %b: !mo.tensor<[4], f32>
) -> !mo.tensor<[D0], f32> {
  %rb = mo.rebind(%b) : !mo.tensor<[4], f32> -> !mo.tensor<[D0], f32>
  %r = map.iter.opaque
      inputs (%a : !mo.tensor<[D0], f32>, %rb as %b1 : !mo.tensor<[D0], f32>)
      outputs (%out : !mo.tensor<[D0], f32>) {
    map.iter.elementwise.body<[D0]>
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[D0], f32>,
                   %b0 = %b1 : !mo.tensor<[D0], f32>,
                   %out0 = %out : !mo.tensor<[D0], f32>} {
      %av = map.load %a0[%idx] : !mo.tensor<[D0], f32>
      %bv = map.load %b0[%idx] : !mo.tensor<[D0], f32>
      %sum = map.call.elementwise @my_add(%idx, %av, %bv)
                : !map.index<1>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[D0], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[D0], f32>
}
