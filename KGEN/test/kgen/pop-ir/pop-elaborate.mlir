// RUN: kgen-opt -elaborate-generators --kgen-print-inline-type-values %s | FileCheck %s

kgen.generator @store_load_pointer(%arg0: i32) -> i32 {
  %0 = pop.stack_allocation 1 x i32
  pop.store %arg0, %0 : !kgen.pointer<i32>
  %1 = pop.stack_allocation 1 x !kgen.pointer<i32>
  pop.store %0, %1 : !kgen.pointer<pointer<i32>>
  %2 = pop.load %1 : !kgen.pointer<pointer<i32>>
  %3 = pop.load %2 : !kgen.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @store_load<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  %0 = pop.stack_allocation 1 x T
  pop.store %arg0, %0 : !kgen.pointer<T>
  %1 = pop.load %0 : !kgen.pointer<T>
  kgen.return %1 : !kgen.paramref<T>
}

kgen.generator @i24_pair_bitcast(%arg0: !pop.array<2, i24>) -> i64 {
  %0 = pop.stack_allocation 2 x i24
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<i24> to !kgen.pointer<array<2, i24>>
  pop.store %arg0, %1 : !kgen.pointer<array<2, i24>>
  %2 = pop.pointer.bitcast %0 : !kgen.pointer<i24> to !kgen.pointer<i64>
  %3 = pop.load %2 : !kgen.pointer<i64>
  kgen.return %3 : i64
}

kgen.generator @bitcast<I: type, O: type>(%arg0: !kgen.paramref<I>) -> !kgen.paramref<O> {
  %0 = pop.stack_allocation 1 x I
  pop.store %arg0, %0 : !kgen.pointer<I>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<I> to !kgen.pointer<O>
  %2 = pop.load %1 : !kgen.pointer<O>
  kgen.return %2 : !kgen.paramref<O>
}

// COM: Store the variant and sneakily read its discriminator's raw value.
kgen.generator @variant_bitcast_discr(%arg0: !kgen.variant<i32, i64>) -> i8 {
  %0 = pop.stack_allocation 1 x !kgen.variant<i32, i64>
  pop.store %arg0, %0 : !kgen.pointer<variant<i32, i64>>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<variant<i32, i64>> to !kgen.pointer<struct<(i64, i8)>>
  %2 = kgen.struct.gep %1[1] : <struct<(i64, i8)>>
  %3 = pop.load %2 : !kgen.pointer<i8>
  kgen.return %3 : i8
}

kgen.generator @array_gep_load<I>(%arg0: !pop.array<3, i24>) -> i24 {
  %0 = kgen.param.constant = <I>
  %1 = pop.stack_allocation 1 x !pop.array<3, i24>
  pop.store %arg0, %1 : !kgen.pointer<array<3, i24>>
  %2 = pop.array.gep %1[%0] : <array<3, i24>>
  %3 = pop.load %2 : !kgen.pointer<i24>
  kgen.return %3 : i24
}

kgen.generator @struct_gep_load(%arg0: !kgen.struct<(i8, i16, i32)>) -> i32 {
  %1 = pop.stack_allocation 1 x !kgen.struct<(i8, i16, i32)>
  pop.store %arg0, %1 : !kgen.pointer<struct<(i8, i16, i32)>>
  %2 = kgen.struct.gep %1[2] : <struct<(i8, i16, i32)>>
  %3 = pop.load %2 : !kgen.pointer<i32>
  kgen.return %3 : i32
}

kgen.generator @bitcast_offset() -> !kgen.struct<(scalar<ui8>, scalar<ui8>)>{
  %x = pop.stack_allocation 1 x !pop.scalar<si64>
  %0 = kgen.param.constant: scalar<si64> = <5>
  pop.store %0, %x : !kgen.pointer<scalar<si64>>
  %1 = pop.pointer.bitcast %x : !kgen.pointer<scalar<si64>> to !kgen.pointer<scalar<ui8>>
  %2 = pop.load %1 : !kgen.pointer<scalar<ui8>>
  %idx1 = index.constant 1
  %3 = pop.offset %1[%idx1] : !kgen.pointer<scalar<ui8>>
  %4 = pop.load %3 : !kgen.pointer<scalar<ui8>>
  %5 = kgen.struct.create(%2, %4) : !kgen.struct<(scalar<ui8>, scalar<ui8>)>
  kgen.return %5 : !kgen.struct<(scalar<ui8>, scalar<ui8>)>
}

kgen.generator @return_heap(%arg0: i16, %arg1: i16) -> !kgen.struct<(pointer<i16>, pointer<i16>)> {
  %idx4 = index.constant 4
  %idx1 = index.constant 1
  %idx32 = index.constant 0x2000
  %0 = pop.aligned_alloc %idx32, %idx4 : <i16>
  pop.store %arg0, %0 : !kgen.pointer<i16>
  %1 = pop.offset %0[%idx1] : !kgen.pointer<i16>
  pop.store %arg1, %1 : !kgen.pointer<i16>
  %2 = kgen.struct.create(%0, %1) : !kgen.struct<(pointer<i16>, pointer<i16>)>
  kgen.return %2 : !kgen.struct<(pointer<i16>, pointer<i16>)>
}

// COM: Check that the pointers alias.
kgen.generator @copy_load(%arg0: !kgen.pointer<i16>, %arg1: !kgen.pointer<i16>) -> i16 {
  %0 = pop.load %arg1: !kgen.pointer<i16>
  %idx1 = index.constant 1
  %1 = pop.offset %arg1[%idx1] : !kgen.pointer<i16>
  pop.store %0, %1 : !kgen.pointer<i16>
  %2 = pop.load %arg0 : !kgen.pointer<i16>
  kgen.return %2 : i16
}

kgen.generator @modify_stack_mem(%arg0: !kgen.pointer<i16>) -> !kgen.pointer<i16> {
  %zero = kgen.param.constant: i16 = <0>
  pop.store %zero, %arg0 : !kgen.pointer<i16>
  kgen.return %arg0 : !kgen.pointer<i16>
}

kgen.generator @return_pointer_to_pointer(%arg0: !kgen.pointer<i16>) -> !kgen.pointer<pointer<i16>> {
  %idx-1 = index.constant -1
  %idx16 = index.constant 16
  %0 = pop.aligned_alloc %idx-1, %idx16 : <pointer<i16>>
  %idx1 = index.constant 1
  %1 = pop.offset %0[%idx1] : !kgen.pointer<pointer<i16>>
  pop.store %arg0, %1 : !kgen.pointer<pointer<i16>>
  kgen.return %1 : !kgen.pointer<pointer<i16>>
}

kgen.generator @load_pointer_to_pointer(%arg0: !kgen.pointer<pointer<i16>>) -> i16 {
  %idx1 = index.constant 1
  %0 = pop.offset %arg0[%idx1] : !kgen.pointer<pointer<i16>>
  %1 = pop.load %0 : !kgen.pointer<pointer<i16>>
  %2 = pop.load %1 : !kgen.pointer<i16>
  kgen.return %2 : i16
}

kgen.generator @free_null(%arg0: !kgen.pointer<i16>) -> index {
  %idx0 = index.constant 0
  pop.aligned_free %arg0 : <i16>
  kgen.return %idx0 : index
}

kgen.generator @freed_memory() -> !kgen.pointer<i16> {
  %idx-1 = index.constant -1
  %idx4 = index.constant 4
  %0 = pop.aligned_alloc %idx-1, %idx4 : <i16>
  %1 = pop.aligned_alloc %idx-1, %idx4 : <i16>
  pop.aligned_free %0 : <i16>
  kgen.return %1 : !kgen.pointer<i16>
}

kgen.generator @const_string(%arg0: !kgen.pointer<i8>) -> !kgen.struct<(i8, pointer<i8>)> {
  %0 = pop.load %arg0 : !kgen.pointer<i8>
  %idx2 = index.constant 2
  %1 = pop.offset %arg0[%idx2] : !kgen.pointer<i8>
  %2 = kgen.struct.create(%0, %1) : !kgen.struct<(i8, pointer<i8>)>
  kgen.return %2 : !kgen.struct<(i8, pointer<i8>)>
}

kgen.generator @parameter_closure() -> index {
  %0 = pop.compiler.global_load "named_global" : index
  kgen.return %0 : index
}

kgen.generator @region_parameter(%arg0: index) -> index {
  pop.compiler.global_store "named_global", %arg0 : index
  %0 = kgen.call @parameter_closure() : () -> index
  kgen.return %0 : index
}

kgen.generator @store_undef() -> index {
  %0 = pop.stack_allocation 1 x index
  %1 = kgen.undef : index
  pop.store %1, %0 : !kgen.pointer<index>
  %2 = pop.load %0 : !kgen.pointer<index>
  kgen.return %2 : index
}

kgen.generator @borrowed_variadic(%arg0: !kgen.pointer<variadic<i32>>, %arg1: index) -> i32{
  %0 = pop.load %arg0 : !kgen.pointer<variadic<i32>>
  %1 = pop.variadic.get %0[%arg1] : !kgen.variadic<i32>
  kgen.return %1 : i32
}

kgen.generator @malloc_and_free(%arg0: i16) -> i16 {
  %idx4 = index.constant 4
  %0 = pop.external_call @malloc(%idx4) : (index) -> (!kgen.pointer<i16>)
  pop.store %arg0, %0 : !kgen.pointer<i16>
  %2 = pop.load %0 : !kgen.pointer<i16>
  pop.external_call @free(%0) : (!kgen.pointer<i16>) -> ()
  kgen.return %2 : i16
}

kgen.generator @variant_discr_gep<Ts: variadic<type>>(%arg0: !kgen.pointer<variant<[Ts]>>) -> !kgen.pointer<scalar<ui8>> {
  %0 = pop.variant.discr_gep %arg0 : <variant<[Ts]>> as <scalar<ui8>>
  kgen.return %0 : !kgen.pointer<scalar<ui8>>
}

// CHECK-LABEL: kgen.func export @do_it
kgen.generator export @do_it() {
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
  kgen.param.constant: struct<(i8, i16, f64)> = <apply(
    :(!kgen.struct<(i8, i16, f64)>) -> !kgen.struct<(i8, i16, f64)> @store_load<:type !kgen.struct<(i8, i16, f64)>>,
    { 120, 32112, 1.125 })>
  // CHECK-NEXT: <{:i32 42, 0}>
  kgen.param.constant: variant<i32, f64> = <apply(
    :(!kgen.variant<i32, f64>) -> !kgen.variant<i32, f64> @store_load<:type !kgen.variant<i32, f64>>,
    {:i32 42, 0})>

  // CHECK-NEXT: <1099511627792>
  kgen.param.constant: i64 = <apply(
    :(!pop.array<2, i24>) -> i64 @i24_pair_bitcast, [16, 256])>

  // CHECK-NEXT: <8590983192>
  kgen.param.constant: i64 = <apply(
    :(!kgen.struct<(i8, i16, i32)>) -> i64 @bitcast<:type !kgen.struct<(i8, i16, i32)>, :type i64>,
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
    :(!kgen.variant<i32, i64>) -> i8 @variant_bitcast_discr, #kgen.variant<:i32 1, 0>)>
  // CHECK-NEXT: <1>
  kgen.param.constant: i8 = <apply(
    :(!kgen.variant<i32, i64>) -> i8 @variant_bitcast_discr, #kgen.variant<:i64 1, 1>)>

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
    :(!kgen.struct<(i8, i16, i32)>) -> i32 @struct_gep_load, { 12, 34, 56 })>

  // CHECK-NEXT: <128>
  kgen.param.constant: i32 = <apply(
    :(!kgen.struct<(i8, i16, i32)>) -> i32 @struct_gep_load, { *?, *?, 128 })>

  // CHECK-NEXT: <{ 5, 0 }>
  kgen.param.constant: struct<(scalar<ui8>, scalar<ui8>)> = <apply(
    :() -> !kgen.struct<(scalar<ui8>, scalar<ui8>)> @bitcast_offset)>

  // CHECK-NEXT: <{ #interp.memref<[([[RETURN_HEAP:.*]], heap, [])], 0, 0>,
  // CHECK-SAME:    #interp.memref<[([[RETURN_HEAP]], heap, [])], 0, 2> }>
  kgen.param.constant: struct<(pointer<i16>, pointer<i16>)> = <apply(
    :(i16, i16) -> !kgen.struct<(pointer<i16>, pointer<i16>)> @return_heap, 0xDEAD, 0xBEEF)>

  // CHECK-NEXT: <-8531>
  kgen.param.constant: i16 = <apply(
    :(!kgen.pointer<i16>, !kgen.pointer<i16>) -> i16 @copy_load,
    #interp.memref<[(mem, heap, [])], 0, 2>, #interp.memref<[(mem, heap, [])], 0, 0>)>

  // CHECK-NEXT: <#interp.memref<[([[MODIFY_STACK_MEM:.*]], stack, [])], 0, 0>>
  kgen.param.constant: pointer<i16> = <apply(
    :(!kgen.pointer<i16>) -> !kgen.pointer<i16> @modify_stack_mem,
    #interp.memref<[(stack, stack, [])], 0, 0>)>

  // CHECK-NEXT: <#interp.memref<[([[RETURN_POINTER:.*]], heap, [(8, 1, 0)]),
  // CHECK-SAME:             ([[RETURN_POINTER_1:.*]], stack, [])], 0, 8>>
  kgen.param.constant: !kgen.pointer<pointer<i16>> = <apply(
    :(!kgen.pointer<i16>) -> !kgen.pointer<pointer<i16>> @return_pointer_to_pointer,
    #interp.memref<[(some_ptr, stack, [])], 0, 0>)>

  // CHECK-NEXT: <-8531>
  kgen.param.constant: i16 = <apply(
    :(!kgen.pointer<pointer<i16>>) -> i16 @load_pointer_to_pointer,
    #interp.memref<[(pointer, stack, [(8, 1, 0)]), (stack, stack, [])], 0, 0>)>

  // CHECK-NEXT: <0>
  kgen.param.constant: index = <apply(
    :(!kgen.pointer<i16>) -> index @free_null, #interp.pointer<0>)>

  // CHECK-NEXT: <#interp.memref<[([[FREED_CONCRETE_MEM:.*]], heap, [])], 0, 0>>
  kgen.param.constant: pointer<i16> = <apply(:() -> !kgen.pointer<i16> @freed_memory)>

  // COM: `ord("hello world"[2]) -> 108`.
  // CHECK-NEXT: <{ 108, #interp.memref<[(string, const_global, [])], 0, 4> }>
  kgen.param.constant: struct<(i8, pointer<i8>)> = <apply(
    :(!kgen.pointer<i8>) -> !kgen.struct<(i8, pointer<i8>)> @const_string,
    #interp.memref<[(string, const_global, [])], 0, 2>)>

  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(index) -> index @region_parameter, 1)>

  // COM: The value is garbage. Just make sure the function interprets.
  // CHECK-NEXT: <{{.*}}>
  kgen.param.constant = <apply(:() -> index @store_undef)>

  // CHECK-NEXT: <tf32>
  kgen.param.constant: dtype = <apply(
    :(!kgen.dtype) -> !kgen.dtype @store_load<:type dtype>, tf32)>

  // CHECK-NEXT: <"hello world">
  kgen.param.constant: string = <apply(
    :(!kgen.string) -> !kgen.string @store_load<:type string>, "hello world")>

  // CHECK-NEXT: <"">
  kgen.param.constant: string = <apply(
    :(!kgen.string) -> !kgen.string @store_load<:type string>, "")>

  // CHECK-NEXT: <[3, 4, 5]>
  kgen.param.constant: variadic<i32> = <apply(
    :(!kgen.variadic<i32>) -> !kgen.variadic<i32> @store_load<:type variadic<i32>>, [3, 4, 5])>

  // CHECK-NEXT: <4>
  kgen.param.constant: i32 = <apply(
    :(!kgen.pointer<variadic<i32>>, index) -> i32 @borrowed_variadic, store_to_mem([3, 4, 5]), 1)>

  // CHECK-NEXT: <[index, {"f" : (i32) -> i32 = @store_load_pointer}]>
  kgen.param.constant: type = <apply(
    :(!kgen.type) -> !kgen.type @store_load<:type type>, [index, {"f" : (i32) -> (i32) = @store_load_pointer}])>

  // CHECK-NEXT: <7>
  kgen.param.constant: i16 = <apply(:(i16) -> i16 @malloc_and_free, 7)>

  // CHECK-NEXT: <8>
  kgen.param.constant: pointer<scalar<ui8>> = <apply(:(!kgen.pointer<variant<i24, i48>>) -> !kgen.pointer<scalar<ui8>>
    @variant_discr_gep<:variadic<type> [i24, i48]>, 0)>
  // CHECK-NEXT: <16>
  kgen.param.constant: pointer<scalar<ui8>> = <apply(:(!kgen.pointer<variant<i24, i48>>) -> !kgen.pointer<scalar<ui8>>
    @variant_discr_gep<:variadic<type> [simd<4, f32>, i8]>, 0)>

  kgen.return
}

// NOTE: Bytes are encoded backwards in resource blobs.
{-#
  // CHECK: dialect_resources
  dialect_resources: {
    // CHECK-NEXT: interp: {
    interp: {
      mem: "0x20000000ADDEEFBE",
      stack: "0x20000000ADDE",
      some_ptr: "0x20000000EFBE",
      pointer: "0x40000000000000000000000000F2052A01000000",
      string: "hello world"
      // CHECK-NEXT: [[RETURN_HEAP]]: "0x00200000ADDEEFBE"
      // CHECK-NEXT: [[MODIFY_STACK_MEM]]: "0x200000000000"
      // COM: 0x77359400 -> 2000000000, the base stack address
      // CHECK-NEXT: [[RETURN_POINTER]]: "0x4000000000000000000000000094357700000000"
      // CHECK-NEXT: [[RETURN_POINTER_1]]: "0x20000000EFBE"
      // CHECK-NEXT: [[FREED_CONCRETE_MEM]]:
      // CHECK-NEXT: string: "hello world"
    // CHECK-NEXT: }
    }
  }
#-}
