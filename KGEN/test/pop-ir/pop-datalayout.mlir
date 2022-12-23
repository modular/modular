// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: @pop_sizeof_alignof
kgen.generator @pop_sizeof_alignof<N, T:type, DT:dtype>() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant = <get_sizeof(!pop.array<1, i8>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  %1 = kgen.param.constant = <get_sizeof(!pop.array<4, i6>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.array<N, i8>, #kgen.target<{{.*}}>)>
  %2 = kgen.param.constant = <get_sizeof(!pop.array<N, i8>, #kgen<target host>)>
  // CHECK-NEXT: <1>
  %3 = kgen.param.constant = <get_alignof(!pop.array<1, i8>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  %4 = kgen.param.constant = <get_alignof(!pop.array<4, i30>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.array<N, i8>, #kgen.target<{{.*}}>)>
  %5 = kgen.param.constant = <get_alignof(!pop.array<N, i8>, #kgen<target host>)>

  // CHECK-NEXT: <8>
  %6 = kgen.param.constant = <get_sizeof(!pop.pointer<scalar<invalid>>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %7 = kgen.param.constant = <get_alignof(!pop.pointer<array<4, i32>>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.pointer<T>, #kgen.target<{{.*}}>)>
  %8 = kgen.param.constant = <get_sizeof(!pop.pointer<T>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.pointer<T>, #kgen.target<{{.*}}>)>
  %9 = kgen.param.constant = <get_alignof(!pop.pointer<T>, #kgen<target host>)>

  // CHECK-NEXT: <4>
  %10 = kgen.param.constant = <get_sizeof(!pop.scalar<si32>, #kgen<target host>)>
  // CHECK-NEXT: <1>
  %11 = kgen.param.constant = <get_alignof(!pop.scalar<si4>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.scalar<DT>, #kgen.target<{{.*}}>)>
  %12 = kgen.param.constant = <get_sizeof(!pop.scalar<DT>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.scalar<DT>, #kgen.target<{{.*}}>)>
  %13 = kgen.param.constant = <get_alignof(!pop.scalar<DT>, #kgen<target host>)>

  // CHECK-NEXT: <16>
  %14 = kgen.param.constant = <get_sizeof(!pop.simd<4, f32>, #kgen<target host>)>
  // CHECK-NEXT: <16>
  %15 = kgen.param.constant = <get_alignof(!pop.simd<4, f32>, #kgen<target host>)>
  // CHECK-NEXT: <get_sizeof(!pop.simd<N, f32>, #kgen.target<{{.*}}>)>
  %16 = kgen.param.constant = <get_sizeof(!pop.simd<N, f32>, #kgen<target host>)>
  // CHECK-NEXT: <get_alignof(!pop.simd<4, DT>, #kgen.target<{{.*}}>)>
  %17 = kgen.param.constant = <get_alignof(!pop.simd<4, DT>, #kgen<target host>)>

  // CHECK-NEXT: <24>
  %18 = kgen.param.constant = <get_sizeof(!pop.struct<i8, i32, i64, i32>, #kgen<target host>)>
  // CHECK-NEXT: <4>
  %19 = kgen.param.constant = <get_alignof(!pop.struct<i8, i32, i16>, #kgen<target host>)>

  // CHECK-NEXT: <16>
  %20 = kgen.param.constant = <get_sizeof(!pop.variant<i32, i16>, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %21 = kgen.param.constant = <get_alignof(!pop.variant<i1, i2, i3, i4>, #kgen<target host>)>

  kgen.return
}
