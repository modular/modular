// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @pop_sizeof_alignof
kgen.generator @pop_sizeof_alignof<N, T:type, DT:dtype>() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <get_sizeof(!pop.array<1, i8>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <get_sizeof(!pop.array<4, i6>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.array<N, i8>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_sizeof(!pop.array<N, i8>, #kgen<target host>)>
  // CHECK-NEXT: <1>
  kgen.param.constant = <get_alignof(!pop.array<1, i8>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <get_alignof(!pop.array<4, i30>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.array<N, i8>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_alignof(!pop.array<N, i8>, #kgen<target host>)>

  // CHECK-NEXT: <8>
  kgen.param.constant = <get_sizeof(!pop.pointer<scalar<invalid>>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  kgen.param.constant = <get_alignof(!pop.pointer<array<4, i32>>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.pointer<T>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_sizeof(!pop.pointer<T>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.pointer<T>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_alignof(!pop.pointer<T>, #kgen<target host>)>

  // CHECK-NEXT: <4>
  kgen.param.constant = <get_sizeof(!pop.scalar<si32>, #kgen<target host>)>
  // CHECK-NEXT: <1>
  kgen.param.constant = <get_alignof(!pop.scalar<si4>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_sizeof(!pop.scalar<DT>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.scalar<DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_alignof(!pop.scalar<DT>, #kgen<target host>)>

  // CHECK-NEXT: <16>
  kgen.param.constant = <get_sizeof(!pop.simd<4, f32>, #kgen<target host>)>
  // CHECK-NEXT: <16>
  kgen.param.constant = <get_alignof(!pop.simd<4, f32>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.simd<N, f32>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_sizeof(!pop.simd<N, f32>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.simd<4, DT>, #kgen.target<{{.*}}>)>
  kgen.param.constant = <get_alignof(!pop.simd<4, DT>, #kgen<target host>)>

  // CHECK-NEXT: <24>
  kgen.param.constant = <get_sizeof(!pop.struct<i8, i32, i64, i32>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <get_alignof(!pop.struct<i8, i32, i16>, #kgen<target host>)>

  // CHECK-NEXT: <16>
  kgen.param.constant = <get_sizeof(!pop.variant<i32, i16>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  kgen.param.constant = <get_alignof(!pop.variant<i1, i2, i3, i4>, #kgen<target host>)>

  // CHECK-NEXT: <16>
  kgen.param.constant = <get_sizeof(!pop.variadic<i32>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  kgen.param.constant = <get_alignof(!pop.variadic<i32>, #kgen<target host>)>

  // CHECK-NEXT: <8>
  kgen.param.constant = <get_sizeof(!pop.coroutine<() -> ()>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  kgen.param.constant = <get_alignof(!pop.coroutine<() -> ()>, #kgen<target host>)>

  kgen.return
}


// CHECK-LABEL: @simd_bitpacked()
kgen.generator @simd_bitpacked() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <get_sizeof(!pop.scalar<si4>, #kgen<target host>)>
  // CHECK-NEXT: <2>
  kgen.param.constant = <get_sizeof(!pop.simd<4, si4>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  kgen.param.constant = <get_sizeof(!pop.scalar<index>, #kgen.target<triple="", cpu="", features="", pointer_bit_width=32, simd_bit_width=128>)>
  // CHECK-NEXT: <8>
  kgen.param.constant = <get_sizeof(!pop.simd<2, address>, #kgen.target<triple="", cpu="", features="", pointer_bit_width=32, simd_bit_width=128>)>
  kgen.return
}
