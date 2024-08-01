// RUN: kgen-opt %s | FileCheck %s

#target = #kgen.target<triple="", arch="", features="", data_layout="i64:64:64", simd_bit_width=128> : !kgen.target
#i32_align8 = #kgen.target<triple="", arch="", features="", data_layout="i32:64:64", simd_bit_width=128> : !kgen.target

// CHECK-LABEL: @pop_sizeof_alignof
kgen.generator @pop_sizeof_alignof<N, T:type, DT:dtype>() {
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(array<1, i8>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(array<4, i6>, #target)>
  // CHECK-NEXT: <get_sizeof(array<N, i8>, #kgen.target<{{.*}}>)>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(array<N, i8>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(array<1, i8>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(array<4, i30>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(array<N, i8>, #target)>

  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(pointer<scalar<invalid>>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(pointer<array<4, i32>>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(pointer<T>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(pointer<T>, #target)>

  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(scalar<si32>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(scalar<si4>, #target)>
  // CHECK-NEXT: <get_sizeof(scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(scalar<DT>, #target)>
  // CHECK-NEXT: <get_alignof(scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: !kgen.int_literal = <get_alignof(scalar<DT>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(simd<4, f32>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_alignof(simd<4, f32>, #target)>
  // CHECK-NEXT: <get_sizeof(simd<N, f32>, #kgen.target<{{.*}}>)>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(simd<N, f32>, #target)>
  // CHECK-NEXT: <get_alignof(simd<4, DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant: !kgen.int_literal = <get_alignof(simd<4, DT>, #target)>

  // CHECK-NEXT: <0>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(struct<()>, #target)>
  // CHECK-NEXT: <24>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(struct<(i8, i32, i64, i32)>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(struct<()>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(struct<(i8, i32, i16)>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(struct<(i32, i8)>, #i32_align8)>

  // CHECK-NEXT: <0>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(!kgen.pack<[]>, #target)>
  // CHECK-NEXT: <24>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(!kgen.pack<[i8, i32, i64, i32]>, #target)>
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(!kgen.pack<[]>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(!kgen.pack<[i8, i32, i16]>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(!kgen.pack<[i32, i8]>, #i32_align8)>

  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(variant<i32, i16>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(variant<i1, i2, i3, i4>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(variadic<i32>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(variadic<i32>, #target)>

  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(union<simd<4, f32>, i64>, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(union<simd<4, f32>, i64>, #target)>

  kgen.return
}

// CHECK-LABEL: @simd_normal()
kgen.generator @simd_normal() {
  // FIXME: get_alignof isn't implemented in terms of DataLayout::getFloatABIAlign.
  // FIXME: get_sizeof doesn't round up to alignment either.
  // https://github.com/modularml/modular/issues/28137

  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_alignof(simd<4, si32>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-v128:256", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(simd<2, si32>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-v64:64-v128:256", simd_bit_width=128>)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-f32:64:64", simd_bit_width=128>)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(f32, #kgen.target<triple="", arch="", features="", data_layout="p:32:32-f32:32:32", simd_bit_width=128>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(scalar<invalid>, #target)>
  kgen.return
}

// CHECK-LABEL: @simd_bitpacked()
kgen.generator @simd_bitpacked() {
  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(scalar<si4>, #target)>
  // CHECK-NEXT: <2>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(simd<4, si4>, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(scalar<index>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(simd<2, address>, #kgen.target<triple="", arch="", features="", data_layout="p:32:32", simd_bit_width=128>)>
  kgen.return
}
