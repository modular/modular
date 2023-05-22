// RUN: kgen-opt -split-input-file -elaborate-generators %s | FileCheck %s

kgen.generator @store_load_pointer(%arg0: i32) -> i32 {
  %0 = pop.stack_allocation 1 x i32
  pop.store %arg0, %0 : !pop.pointer<i32>
  %1 = pop.stack_allocation 1 x !pop.pointer<i32>
  pop.store %0, %1 : !pop.pointer<pointer<i32>>
  %2 = pop.load %1 : !pop.pointer<pointer<i32>>
  %3 = pop.load %2 : !pop.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @store_load<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  %0 = pop.stack_allocation 1 x T
  pop.store %arg0, %0 : !pop.pointer<T>
  %1 = pop.load %0 : !pop.pointer<T>
  kgen.return %1 : !kgen.paramref<T>
}

kgen.generator @i24_pair_bitcast(%arg0: !pop.array<2, i24>) -> i64 {
  %0 = pop.stack_allocation 2 x i24
  %1 = pop.pointer.bitcast %0 : !pop.pointer<i24> to !pop.pointer<array<2, i24>>
  pop.store %arg0, %1 : !pop.pointer<array<2, i24>>
  %2 = pop.pointer.bitcast %0 : !pop.pointer<i24> to !pop.pointer<i64>
  %3 = pop.load %2 : !pop.pointer<i64>
  kgen.return %3 : i64
}

kgen.generator @bitcast<I: type, O: type>(%arg0: !kgen.paramref<I>) -> !kgen.paramref<O> {
  %0 = pop.stack_allocation 1 x I
  pop.store %arg0, %0 : !pop.pointer<I>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<I> to !pop.pointer<O>
  %2 = pop.load %1 : !pop.pointer<O>
  kgen.return %2 : !kgen.paramref<O>
}

// COM: Store the variant and sneakily read its discriminator's raw value.
kgen.generator @variant_bitcast_discr(%arg0: !pop.variant<i32, i64>) -> i8 {
  %0 = pop.stack_allocation 1 x !pop.variant<i32, i64>
  pop.store %arg0, %0 : !pop.pointer<variant<i32, i64>>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<variant<i32, i64>> to !pop.pointer<struct<i64, i8>>
  %2 = pop.struct.gep %1[1] : <struct<i64, i8>>
  %3 = pop.load %2 : !pop.pointer<i8>
  kgen.return %3 : i8
}

kgen.generator @array_gep_load<I>(%arg0: !pop.array<3, i24>) -> i24 {
  %0 = kgen.param.constant = <I>
  %1 = pop.stack_allocation 1 x !pop.array<3, i24>
  pop.store %arg0, %1 : !pop.pointer<array<3, i24>>
  %2 = pop.array.gep %1[%0] : <array<3, i24>>
  %3 = pop.load %2 : !pop.pointer<i24>
  kgen.return %3 : i24
}

kgen.generator @struct_gep_load(%arg0: !pop.struct<i8, i16, i32>) -> i32 {
  %1 = pop.stack_allocation 1 x !pop.struct<i8, i16, i32>
  pop.store %arg0, %1 : !pop.pointer<struct<i8, i16, i32>>
  %2 = pop.struct.gep %1[2] : <struct<i8, i16, i32>>
  %3 = pop.load %2 : !pop.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @bitcast_offset() -> !pop.struct<scalar<ui8>, scalar<ui8>>{
  %x = pop.stack_allocation 1 x !pop.scalar<si64>
  %0 = kgen.param.constant: scalar<si64> = <5>
  pop.store %0, %x : !pop.pointer<scalar<si64>>
  %1 = pop.pointer.bitcast %x : !pop.pointer<scalar<si64>> to !pop.pointer<scalar<ui8>>
  %2 = pop.load %1 : !pop.pointer<scalar<ui8>>
  %idx1 = index.constant 1
  %3 = pop.offset %1[%idx1] : !pop.pointer<scalar<ui8>>
  %4 = pop.load %3 : !pop.pointer<scalar<ui8>>
  %5 = pop.struct.create(%2, %4) : !pop.struct<scalar<ui8>, scalar<ui8>>
  kgen.return %5 : !pop.struct<scalar<ui8>, scalar<ui8>>
}

kgen.export @do_it

// CHECK-LABEL: kgen.func @do_it
kgen.generator @do_it() {
  // CHECK-NEXT: <555>
  kgen.param.constant: i32 = <apply(
    :(i32) -> i32 @store_load_pointer, 555)>

  // CHECK-NEXT: [123, 456]
  kgen.param.constant: array<2, i24> = <apply(
    :(!pop.array<2, i24>) -> !pop.array<2, i24> @store_load<:type !pop.array<2, i24>>,
    [123, 456])>
  // CHECK-NEXT: <"1.25", "2.25">
  kgen.param.constant: simd<2, f32> = <apply(
    :(!pop.simd<2, f32>) -> !pop.simd<2, f32> @store_load<:type !pop.simd<2, f32>>,
    <"1.25", "2.25">)>
  // CHECK-NEXT: <-7, 7>
  kgen.param.constant: simd<2, si4> = <apply(
    :(!pop.simd<2, si4>) -> !pop.simd<2, si4> @store_load<:type !pop.simd<2, si4>>,
    <-7, 7>)>
  // CHECK-NEXT: <0, 1, 2, 3, 3, 2>
  kgen.param.constant: simd<6, ui2> = <apply(
    :(!pop.simd<6, ui2>) -> !pop.simd<6, ui2> @store_load<:type !pop.simd<6, ui2>>,
    <0, 1, 2, 3, 3, 2>)>
  // CHECK-NEXT: <-5>
  kgen.param.constant: scalar<index> = <apply(
    :(!pop.scalar<index>) -> !pop.scalar<index> @store_load<:type !pop.scalar<index>>,
    <-5>)>
  // CHECK-NEXT: { 120, 32112, 1.125{{0+}}e+00 }
  kgen.param.constant: struct<i8, i16, f64> = <apply(
    :(!pop.struct<i8, i16, f64>) -> !pop.struct<i8, i16, f64> @store_load<:type !pop.struct<i8, i16, f64>>,
    { 120, 32112, 1.125 })>
  // CHECK-NEXT: #pop.variant<:i32 42>
  kgen.param.constant: variant<i32, f64> = <apply(
    :(!pop.variant<i32, f64>) -> !pop.variant<i32, f64> @store_load<:type !pop.variant<i32, f64>>,
    #pop.variant<:i32 42>)>

  // CHECK-NEXT: <1099511627792>
  kgen.param.constant: i64 = <apply(
    :(!pop.array<2, i24>) -> i64 @i24_pair_bitcast, [16, 256])>

  // CHECK-NEXT: <8590983192>
  kgen.param.constant: i64 = <apply(
    :(!pop.struct<i8, i16, i32>) -> i64 @bitcast<:type !pop.struct<i8, i16, i32>, :type i64>,
    { 24, 16, 2 })>
  // CHECK-NEXT: <1026>
  kgen.param.constant: i16 = <apply(
    :(!pop.simd<2, si8>) -> i16 @bitcast<:type !pop.simd<2, si8>, :type i16>,
    <2, 4>)>
  // CHECK-NEXT: <229>
  kgen.param.constant: ui8 = <apply(
    :(!pop.simd<4, ui2>) -> ui8 @bitcast<:type !pop.simd<4, ui2>, :type ui8>,
    <1, 1, 2, 3>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: i8 = <apply(
    :(!pop.variant<i32, i64>) -> i8 @variant_bitcast_discr, #pop.variant<:i32 1>)>
  // CHECK-NEXT: <1>
  kgen.param.constant: i8 = <apply(
    :(!pop.variant<i32, i64>) -> i8 @variant_bitcast_discr, #pop.variant<:i64 1>)>

  // CHECK-NEXT: <12>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<0>, [12, 34, 56])>
  // CHECK-NEXT: <34>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<1>, [12, 34, 56])>
  // CHECK-NEXT: <56>
  kgen.param.constant: i24 = <apply(
    :(!pop.array<3, i24>) -> i24 @array_gep_load<2>, [12, 34, 56])>

  // CHECK-NEXT: <56>
  kgen.param.constant: i32 = <apply(
    :(!pop.struct<i8, i16, i32>) -> i32 @struct_gep_load, { 12, 34, 56 })>

  // CHECK-NEXT: <{ 5, 0 }>
  kgen.param.constant: struct<scalar<ui8>, scalar<ui8>> = <apply(
    :() -> !pop.struct<scalar<ui8>, scalar<ui8>> @bitcast_offset)>

  kgen.return
}
