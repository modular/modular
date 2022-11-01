// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @pop_sizeof_alignof
kgen.generator @pop_sizeof_alignof<N, T:type, DT:dtype>() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <get_sizeof(!pop.array<1, i8>)>
  // CHECK-NEXT: <4>
  %1 = kgen.param.constant = <get_sizeof(!pop.array<4, i6>)>
  // CHECK-NEXT: <get_sizeof(!pop.array<N, i8>)>
  %2 = kgen.param.constant = <get_sizeof(!pop.array<N, i8>)>
  // CHECK-NEXT: <1>
  %3 = kgen.param.constant = <get_alignof(!pop.array<1, i8>)>
  // CHECK-NEXT: <4>
  %4 = kgen.param.constant = <get_alignof(!pop.array<4, i30>)>
  // CHECK-NEXT: <get_alignof(!pop.array<N, i8>)>
  %5 = kgen.param.constant = <get_alignof(!pop.array<N, i8>)>

  // CHECK-NEXT: <8>
  %6 = kgen.param.constant = <get_sizeof(!pop.pointer<scalar<invalid>>)>
  // CHECK-NEXT: <8>
  %7 = kgen.param.constant = <get_alignof(!pop.pointer<array<4, i32>>)>
  // CHECK-NEXT: <get_sizeof(!pop.pointer<T>)>
  %8 = kgen.param.constant = <get_sizeof(!pop.pointer<T>)>
  // CHECK-NEXT: <get_alignof(!pop.pointer<T>)>
  %9 = kgen.param.constant = <get_alignof(!pop.pointer<T>)>

  // CHECK-NEXT: <4>
  %10 = kgen.param.constant = <get_sizeof(!pop.scalar<si32>)>
  // CHECK-NEXT: <1>
  %11 = kgen.param.constant = <get_alignof(!pop.scalar<si4>)>
  // CHECK-NEXT: <get_sizeof(!pop.scalar<DT>)>
  %12 = kgen.param.constant = <get_sizeof(!pop.scalar<DT>)>
  // CHECK-NEXT: <get_alignof(!pop.scalar<DT>)>
  %13 = kgen.param.constant = <get_alignof(!pop.scalar<DT>)>

  // CHECK-NEXT: <16>
  %14 = kgen.param.constant = <get_sizeof(!pop.simd<4, f32>)>
  // CHECK-NEXT: <16>
  %15 = kgen.param.constant = <get_alignof(!pop.simd<4, f32>)>
  // CHECK-NEXT: <get_sizeof(!pop.simd<N, f32>)>
  %16 = kgen.param.constant = <get_sizeof(!pop.simd<N, f32>)>
  // CHECK-NEXT: <get_alignof(!pop.simd<4, DT>)>
  %17 = kgen.param.constant = <get_alignof(!pop.simd<4, DT>)>

  // CHECK-NEXT: <24>
  %18 = kgen.param.constant = <get_sizeof(!pop.struct<i8, i32, i64, i32>)>
  // CHECK-NEXT: <4>
  %19 = kgen.param.constant = <get_alignof(!pop.struct<i8, i32, i16>)>

  kgen.return
}
