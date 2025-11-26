// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @external_call
  kgen.func @external_call(%a: !pop.simd<1, ui32>, %b: !kgen.pointer<i32>) -> !pop.simd<4, f64> {
    // CHECK: llvm.call @foo
    %0 = pop.external_call @foo(%a) attributes {
      funcAttrs = ["noinline", "noreturn"],
      memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = none, targetMem0 = none, targetMem1 = none>
    } : (!pop.simd<1, ui32>) -> !pop.simd<4, f64>
    // CHECK: llvm.call @bar
    %1 = pop.external_call @bar(%b) attributes {argAttrs = [{llvm.noalias}], resAttrs = [{llvm.signext}]} : (!kgen.pointer<i32>) -> i32
    kgen.return %0 : !pop.simd<4, f64>
  }
  // CHECK: llvm.func @foo(i32) -> vector<4xf64>
  // CHECK-SAME: memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read, errnoMem = none, targetMem0 = none, targetMem1 = none>
  // CHECK-SAME: passthrough = ["noinline", "noreturn"
  // CHECK: llvm.func @bar(!llvm.ptr {llvm.noalias}) -> (i32 {llvm.signext})
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @external_call_variadic
  kgen.func @external_call_variadic(%a: !pop.simd<1, ui32>) {
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a) (!pop.simd<1, ui32>) -> () : (!pop.simd<1, ui32>) -> ()
    // CHECK: llvm.call @foo
    pop.external_call @foo (%a, %a) (!pop.simd<1, ui32>) -> () : (!pop.simd<1, ui32>, !pop.simd<1, ui32>) -> ()
    kgen.return
  }
  // CHECK: llvm.func @foo(i32, ...)
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_constant
  kgen.func @global_constant() {
    // CHECK: llvm.mlir.addressof @global_constant_0 : !llvm.ptr
    %0 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_0 : !llvm.ptr
    %1 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_1 : !llvm.ptr
    %2 = pop.global_constant: simd<2, si32> = <<2, 5>>
    kgen.return
  }

  // CHECK-LABEL: kgen.func @global_alloc
  kgen.func @global_alloc() -> !kgen.pointer<scalar<f32>, 3> {
    // CHECK-NEXT: %0 = llvm.mlir.addressof @my_alloc : !llvm.ptr<3>
    // CHECK-NEXT: %1 = llvm.bitcast %0 : !llvm.ptr<3> to !llvm.ptr<3>
    %0 = pop.global_alloc "my_alloc" 2 x !pop.scalar<f32> address_space 3 align 4
    kgen.return %0 : !kgen.pointer<scalar<f32>, 3>
  }

  // CHECK-LABEL: llvm.mlir.global internal @my_alloc() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x f32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @aligned_globals
  kgen.func @aligned_globals() {

    // Same value + same alignment = same constant.
    // CHECK: llvm.mlir.addressof @global_constant
    %0 = pop.global_constant: ui32 = <5> align 4
    // CHECK: llvm.mlir.addressof @global_constant
    %1 = pop.global_constant: ui32 = <5> align 4

    // Same value + different alignment = different constant.
    // CHECK: llvm.mlir.addressof @global_constant_0
    %2 = pop.global_constant: ui32 = <5> align 16

    // CHECK: llvm.mlir.addressof @global_constant_1
    %3 = pop.global_constant: simd<2, si32> = <<2, 5>> align 64
    kgen.return
  }

  // CHECK: llvm.mlir.global internal constant @global_constant() {addr_space = 0 : i32, alignment = 4 : i64} : i32 {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(5 : i32) : i32

  // CHECK: llvm.mlir.global internal constant @global_constant_0() {addr_space = 0 : i32, alignment = 16 : i64} : i32 {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(5 : i32) : i32

  // CHECK: llvm.mlir.global internal constant @global_constant_1() {addr_space = 0 : i32, alignment = 64 : i64} : vector<2xi32>
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(#M.dense_array<2, 5> : vector<2xi32>) : vector<2xi32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_array_constant
  kgen.func @global_array_constant() {
    // CHECK: llvm.mlir.addressof @global_constant
    %0 = pop.global_constant: array<4, ui32> = <[1, 2, 3, 4]>
    kgen.return
  }
  // CHECK: llvm.mlir.global internal constant @global_constant() {
  // CHECK: %0 = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK: llvm.return %{{.*}} : !llvm.array<4 x i32>
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: llvm.func @alloc_free
  llvm.func @alloc_free() {
    %size = index.constant 1
    %align = index.constant 8
    // CHECK: [[RAW_PTR:%.*]] = llvm.call @KGEN_CompilerRT_AlignedAlloc
    // CHECK-NEXT: [[PTR:%.*]] = llvm.bitcast [[RAW_PTR]] : !llvm.ptr to !llvm.ptr
    %0 = pop.aligned_alloc %align, %size : <index>
    // CHECK: [[RAW_PTR:%.*]] = llvm.bitcast [[PTR]] : !llvm.ptr to !llvm.ptr
    // CHECK-NEXT: llvm.call @KGEN_CompilerRT_AlignedFree([[RAW_PTR]])
    pop.aligned_free %0 : <index>
    llvm.return
  }

  // CHECK: llvm.func @KGEN_CompilerRT_AlignedAlloc(i64 {llvm.allocalign}, i64) -> (!llvm.ptr {llvm.noalias})
  // CHECK-DAG: ["allockind", "41"]
  // CHECK-DAG: ["allocsize", "8589934591"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]

  // CHECK: llvm.func @KGEN_CompilerRT_AlignedFree(!llvm.ptr {llvm.allocptr})
  // CHECK-DAG: ["allockind", "4"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @external_call
kgen.func @external_call(%a : !kgen.struct<(scalar<si32>)>,
                         %b : !kgen.struct<(scalar<si32>, scalar<f32>, scalar<f32>)>) {
   // CHECK: %0 = builtin.unrealized_conversion_cast %arg1
   // CHECK: %1 = builtin.unrealized_conversion_cast %arg0

   // CHECK: %2 = llvm.extractvalue %1[0]
   // CHECK: llvm.call @call1(%2) : (i32) -> i32
   %0 = pop.external_call @call1(%a) : (!kgen.struct<(scalar<si32>)>) -> !pop.scalar<si32>

   // CHECK: %4 = llvm.extractvalue %0[0]
   // CHECK: %5 = llvm.extractvalue %0[1]
   // CHECK: %6 = llvm.extractvalue %0[2]
   // CHECK: %7 = llvm.call @call3(%4, %5, %6) : (i32, f32, f32) -> i32
   %1 = pop.external_call @call3(%b) : (!kgen.struct<(scalar<si32>, scalar<f32>, scalar<f32>)>) -> !pop.scalar<si32>
   kgen.return
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @external_call_pack
kgen.func @external_call_pack() {
   // CHECK: [[R:%.*]] = llvm.call @call1() : () -> !llvm.struct<(i32, i64)>
   // CHECK-NEXT: llvm.extractvalue [[R]][0]
   // CHECK-NEXT: llvm.extractvalue [[R]][1]
   %0:2 = pop.external_call @call1() : () -> (i32, i64)
   kgen.return
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @noalias_cast
kgen.func @noalias_cast(%arg0: !kgen.pointer<index>) -> index {
  // CHECK: llvm.call @__kgen_noalias_cast(%0)
  %0 = pop.noalias_pointer_cast %arg0 : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK: llvm.func internal @__kgen_noalias_cast
// CHECK-SAME: (%arg0: !llvm.ptr {llvm.noalias}) -> (!llvm.ptr {llvm.noalias})
// CHECK-SAME: ["alwaysinline", "mustprogress", "nofree", "norecurse", "nosync",
// CHECK-SAME:  "nounwind", "willreturn", ["memory", "0"]
// CHECK-NEXT: return %arg0

}

// -----

module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "", data_layout = "", simd_bit_width = 128>} {
  // CHECK-LABEL: @test_air_cos_f16_mangling
  llvm.func @test_air_cos_f16_mangling(%arg0: !llvm.struct<(f16)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f16)> to !kgen.struct<(scalar<f16>)>
    // CHECK: llvm.call @air.cos.f16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<f16>)>) -> !pop.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_f32_mangling
  llvm.func @test_air_cos_f32_mangling(%arg0: !llvm.struct<(f32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f32)> to !kgen.struct<(scalar<f32>)>
    // CHECK: llvm.call @air.cos.f32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<f32>)>) -> !pop.scalar<f32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_bf16_mangling
  llvm.func @test_air_cos_bf16_mangling(%arg0: !llvm.struct<(bf16)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(bf16)> to !kgen.struct<(scalar<bf16>)>
    // CHECK: llvm.call @air.cos.bf16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<bf16>)>) -> !pop.scalar<bf16>
    llvm.return
  }

  // COM: There's no cos for integer type, but it's needed to verify mangling is correct
  // CHECK-LABEL: @test_air_cos_i32_mangling
  llvm.func @test_air_cos_i32_mangling(%arg0: !llvm.struct<(i32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i32)> to !kgen.struct<(scalar<si32>)>
    // CHECK: llvm.call @air.cos.i32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<si32>)>) -> !pop.scalar<si32>
    llvm.return
  }

  // COM: There's no cos for integer type, but it's needed to verify mangling is correct
  // CHECK-LABEL: @test_air_cos_i128_mangling
  llvm.func @test_air_cos_i128_mangling(%arg0: !llvm.struct<(i128)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i128)> to !kgen.struct<(scalar<si128>)>
    // CHECK: llvm.call @air.cos.i128
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<si128>)>) -> !pop.scalar<si128>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_bf16_mangling
  kgen.func @test_air_max_bf16_mangling(%arg0: !pop.scalar<bf16>, %arg1: !pop.scalar<bf16>) {
    // CHECK: llvm.call @air.fmax.bf16
    %0 = pop.max %arg0, %arg1 : !pop.scalar<bf16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_v2f16_mangling
  llvm.func @test_air_cos_v2f16_mangling(%arg0: !llvm.struct<(vector<2 x f16>)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(vector<2 x f16>)> to !kgen.struct<(!pop.simd<2, f16>)>
    // CHECK: llvm.call @air.cos.v2f16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(!pop.simd<2, f16>)>) -> !pop.simd<2, f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_f16_mangling
  kgen.func @test_air_min_f16_mangling(%arg0: !pop.scalar<f16>, %arg1: !pop.scalar<f16>) {
    // CHECK: llvm.call @air.fmin.f16
    %0 = pop.min %arg0, %arg1 : !pop.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_si32_mangling
  kgen.func @test_air_min_si32_mangling(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) {
    // CHECK: llvm.call @air.min.s.i32
    %0 = pop.min %arg0, %arg1 : !pop.scalar<si32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_ui32_mangling
  kgen.func @test_air_min_ui32_mangling(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) {
    // CHECK: llvm.call @air.min.u.i32
    %0 = pop.min %arg0, %arg1 : !pop.scalar<ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_v2i32_mangling
  kgen.func @test_air_min_v2i32_mangling(%arg0: !pop.simd<2, ui32>, %arg1: !pop.simd<2, ui32>) {
    // CHECK: llvm.call @air.min.u.v2i32
    %0 = pop.min %arg0, %arg1 : !pop.simd<2, ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_f16_mangling
  kgen.func @test_air_max_f16_mangling(%arg0: !pop.scalar<f16>, %arg1: !pop.scalar<f16>) {
    // CHECK: llvm.call @air.fmax.f16
    %0 = pop.max %arg0, %arg1 : !pop.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_si32_mangling
  kgen.func @test_air_max_si32_mangling(%arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>) {
    // CHECK: llvm.call @air.max.s.i32
    %0 = pop.max %arg0, %arg1 : !pop.scalar<si32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_ui32_mangling
  kgen.func @test_air_max_ui32_mangling(%arg0: !pop.scalar<ui32>, %arg1: !pop.scalar<ui32>) {
    // CHECK: llvm.call @air.max.u.i32
    %0 = pop.max %arg0, %arg1 : !pop.scalar<ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_v2i32_mangling
  kgen.func @test_air_max_v2i32_mangling(%arg0: !pop.simd<2, ui32>, %arg1: !pop.simd<2, ui32>) {
    // CHECK: llvm.call @air.max.u.v2i32
    %0 = pop.max %arg0, %arg1 : !pop.simd<2, ui32>
    llvm.return
  }

  kgen.func @test_pop_cast_all_types(
    %si8_val: !pop.scalar<si8>,
    %ui8_val: !pop.scalar<ui8>,
    %si16_val: !pop.scalar<si16>,
    %ui16_val: !pop.scalar<ui16>,
    %si32_val: !pop.scalar<si32>,
    %ui32_val: !pop.scalar<ui32>,
    %f16_val: !pop.scalar<f16>,
    %bf16_val: !pop.scalar<bf16>,
    %f32_val: !pop.scalar<f32>
  ) {
    // CHECK-LABEL:   kgen.func @test_pop_cast_all_types(
    // CHECK-SAME: %[[ARG0:.*]]: !pop.scalar<si8>,
    // CHECK-SAME: %[[ARG1:.*]]: !pop.scalar<ui8>,
    // CHECK-SAME: %[[ARG2:.*]]: !pop.scalar<si16>,
    // CHECK-SAME: %[[ARG3:.*]]: !pop.scalar<ui16>,
    // CHECK-SAME: %[[ARG4:.*]]: !pop.scalar<si32>,
    // CHECK-SAME: %[[ARG5:.*]]: !pop.scalar<ui32>,
    // CHECK-SAME: %[[ARG6:.*]]: !pop.scalar<f16>,
    // CHECK-SAME: %[[ARG7:.*]]: !pop.scalar<bf16>,
    // CHECK-SAME: %[[ARG8:.*]]: !pop.scalar<f32>) {
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG8]] : !pop.scalar<f32> to f32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG7]] : !pop.scalar<bf16> to bf16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ARG6]] : !pop.scalar<f16> to f16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[ARG5]] : !pop.scalar<ui32> to i32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_4:.*]] = builtin.unrealized_conversion_cast %[[ARG4]] : !pop.scalar<si32> to i32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_5:.*]] = builtin.unrealized_conversion_cast %[[ARG3]] : !pop.scalar<ui16> to i16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_6:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : !pop.scalar<si16> to i16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_7:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !pop.scalar<ui8> to i8
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_8:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !pop.scalar<si8> to i8
    // ========================================================================
    // from si8
    // ========================================================================
    // CHECK: %[[CALL_0:.*]] = llvm.call @air.convert.f.f16.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> f16
    %si8_to_f16 = pop.cast %si8_val : !pop.scalar<si8> to !pop.scalar<f16>

    // CHECK: %[[CALL_1:.*]] = llvm.call @air.convert.f.bf16.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> bf16
    %si8_to_bf16 = pop.cast %si8_val : !pop.scalar<si8> to !pop.scalar<bf16>

    // CHECK: %[[CALL_2:.*]] = llvm.call @air.convert.f.f32.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> f32
    %si8_to_f32 = pop.cast %si8_val : !pop.scalar<si8> to !pop.scalar<f32>

    // ========================================================================
    // from ui8
    // ========================================================================
    // CHECK: %[[CALL_3:.*]] = llvm.call @air.convert.f.f16.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> f16
    %ui8_to_f16 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<f16>

    // CHECK: %[[CALL_4:.*]] = llvm.call @air.convert.f.bf16.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> bf16
    %ui8_to_bf16 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<bf16>

    // CHECK: %[[CALL_5:.*]] = llvm.call @air.convert.f.f32.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> f32
    %ui8_to_f32 = pop.cast %ui8_val : !pop.scalar<ui8> to !pop.scalar<f32>

    // ========================================================================
    // from si16
    // ========================================================================
    // CHECK: %[[CALL_6:.*]] = llvm.call @air.convert.f.f16.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> f16
    %si16_to_f16 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<f16>

    // CHECK: %[[CALL_7:.*]] = llvm.call @air.convert.f.bf16.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> bf16
    %si16_to_bf16 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<bf16>

    // CHECK: %[[CALL_8:.*]] = llvm.call @air.convert.f.f32.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> f32
    %si16_to_f32 = pop.cast %si16_val : !pop.scalar<si16> to !pop.scalar<f32>

    // ========================================================================
    // from ui16
    // ========================================================================
    // CHECK: %[[CALL_9:.*]] = llvm.call @air.convert.f.f16.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> f16
    %ui16_to_f16 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<f16>

    // CHECK: %[[CALL_10:.*]] = llvm.call @air.convert.f.bf16.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> bf16
    %ui16_to_bf16 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<bf16>

    // CHECK: %[[CALL_11:.*]] = llvm.call @air.convert.f.f32.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> f32
    %ui16_to_f32 = pop.cast %ui16_val : !pop.scalar<ui16> to !pop.scalar<f32>

    // ========================================================================
    // from si32
    // ========================================================================
    // CHECK: %[[CALL_12:.*]] = llvm.call @air.convert.f.f16.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> f16
    %si32_to_f16 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<f16>

    // CHECK: %[[CALL_13:.*]] = llvm.call @air.convert.f.bf16.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> bf16
    %si32_to_bf16 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<bf16>

    // CHECK: %[[CALL_14:.*]] = llvm.call @air.convert.f.f32.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> f32
    %si32_to_f32 = pop.cast %si32_val : !pop.scalar<si32> to !pop.scalar<f32>

    // ========================================================================
    // from ui32
    // ========================================================================
    // CHECK: %[[CALL_15:.*]] = llvm.call @air.convert.f.f16.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> f16
    %ui32_to_f16 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<f16>

    // CHECK: %[[CALL_16:.*]] = llvm.call @air.convert.f.bf16.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> bf16
    %ui32_to_bf16 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<bf16>

    // CHECK: %[[CALL_17:.*]] = llvm.call @air.convert.f.f32.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> f32
    %ui32_to_f32 = pop.cast %ui32_val : !pop.scalar<ui32> to !pop.scalar<f32>

    // ========================================================================
    // from f16
    // ========================================================================
    // CHECK: %[[CALL_18:.*]] = llvm.call @air.convert.s.i8.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i8
    %f16_to_si8 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<si8>

    // CHECK: %[[CALL_19:.*]] = llvm.call @air.convert.u.i8.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i8
    %f16_to_ui8 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<ui8>

    // CHECK: %[[CALL_20:.*]] = llvm.call @air.convert.s.i16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i16
    %f16_to_si16 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<si16>

    // CHECK: %[[CALL_21:.*]] = llvm.call @air.convert.u.i16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i16
    %f16_to_ui16 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<ui16>

    // CHECK: %[[CALL_22:.*]] = llvm.call @air.convert.s.i32.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i32
    %f16_to_si32 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<si32>

    // CHECK: %[[CALL_23:.*]] = llvm.call @air.convert.u.i32.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i32
    %f16_to_ui32 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<ui32>

    // CHECK: %[[CALL_24:.*]] = llvm.call @air.convert.f.bf16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> bf16
    %f16_to_bf16 = pop.cast %f16_val : !pop.scalar<f16> to !pop.scalar<bf16>

    // ========================================================================
    // from bf16
    // ========================================================================
    // CHECK: %[[CALL_24:.*]] = llvm.call @air.convert.s.i8.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i8
    %bf16_to_si8 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<si8>

    // CHECK: %[[CALL_25:.*]] = llvm.call @air.convert.u.i8.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i8
    %bf16_to_ui8 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<ui8>

    // CHECK: %[[CALL_26:.*]] = llvm.call @air.convert.s.i16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i16
    %bf16_to_si16 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<si16>

    // CHECK: %[[CALL_27:.*]] = llvm.call @air.convert.u.i16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i16
    %bf16_to_ui16 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<ui16>

    // CHECK: %[[CALL_28:.*]] = llvm.call @air.convert.s.i32.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i32
    %bf16_to_si32 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<si32>

    // CHECK: %[[CALL_29:.*]] = llvm.call @air.convert.u.i32.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i32
    %bf16_to_ui32 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<ui32>

    // CHECK: %[[CALL_291:.*]] = llvm.call @air.convert.f.f16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> f16
    %bf16_to_f16 = pop.cast %bf16_val : !pop.scalar<bf16> to !pop.scalar<f16>

    // ========================================================================
    // from f32
    // ========================================================================
    // CHECK: %[[CALL_30:.*]] = llvm.call @air.convert.s.i8.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i8
    %f32_to_si8 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<si8>

    // CHECK: %[[CALL_31:.*]] = llvm.call @air.convert.u.i8.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i8
    %f32_to_ui8 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<ui8>

    // CHECK: %[[CALL_32:.*]] = llvm.call @air.convert.s.i16.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i16
    %f32_to_si16 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<si16>

    // CHECK: %[[CALL_33:.*]] = llvm.call @air.convert.u.i16.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i16
    %f32_to_ui16 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<ui16>

    // CHECK: %[[CALL_34:.*]] = llvm.call @air.convert.s.i32.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i32
    %f32_to_si32 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<si32>

    // CHECK: %[[CALL_35:.*]] = llvm.call @air.convert.u.i32.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i32
    %f32_to_ui32 = pop.cast %f32_val : !pop.scalar<f32> to !pop.scalar<ui32>

    kgen.return
  }

  // CHECK-LABEL:   @test_vector_pop_cast_all_types(
  kgen.func @test_vector_pop_cast_all_types(
    %si8_val: !pop.simd<2, si8>,
    %ui8_val: !pop.simd<2, ui8>,
    %si16_val: !pop.simd<2, si16>,
    %ui16_val: !pop.simd<2, ui16>,
    %si32_val: !pop.simd<2, si32>,
    %ui32_val: !pop.simd<2, ui32>,
    %f16_val: !pop.simd<2, f16>,
    %bf16_val: !pop.simd<2, bf16>,
    %f32_val: !pop.simd<2, f32>
  ) {
    // CHECK:  %[[CALL_0:.*]] = llvm.call @air.convert.s.v2i16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_1:.*]] = llvm.call @air.convert.u.v2i16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_2:.*]] = llvm.call @air.convert.s.v2i32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_3:.*]] = llvm.call @air.convert.u.v2i32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_4:.*]] = llvm.call @air.convert.f.v2f16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf16>
    // CHECK:  %[[CALL_5:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xbf16>
    // CHECK:  %[[CALL_6:.*]] = llvm.call @air.convert.f.v2f32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf32>
    // CHECK:  %[[CALL_7:.*]] = llvm.call @air.convert.s.v2i16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_8:.*]] = llvm.call @air.convert.u.v2i16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_9:.*]] = llvm.call @air.convert.s.v2i32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_10:.*]] = llvm.call @air.convert.u.v2i32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_11:.*]] = llvm.call @air.convert.f.v2f16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf16>
    // CHECK:  %[[CALL_12:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xbf16>
    // CHECK:  %[[CALL_13:.*]] = llvm.call @air.convert.f.v2f32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf32>
    // CHECK:  %[[CALL_14:.*]] = llvm.call @air.convert.s.v2i8.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_15:.*]] = llvm.call @air.convert.u.v2i8.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_16:.*]] = llvm.call @air.convert.s.v2i32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_17:.*]] = llvm.call @air.convert.u.v2i32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_18:.*]] = llvm.call @air.convert.f.v2f16.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf16>
    // CHECK:  %[[CALL_19:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_20:.*]] = llvm.call @air.convert.f.v2f32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf32>
    // CHECK:  %[[CALL_21:.*]] = llvm.call @air.convert.s.v2i8.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_22:.*]] = llvm.call @air.convert.u.v2i8.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_23:.*]] = llvm.call @air.convert.s.v2i32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_24:.*]] = llvm.call @air.convert.u.v2i32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_25:.*]] = llvm.call @air.convert.f.v2f16.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf16>
    // CHECK:  %[[CALL_26:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_27:.*]] = llvm.call @air.convert.f.v2f32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf32>
    // CHECK:  %[[CALL_28:.*]] = llvm.call @air.convert.s.v2i8.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_29:.*]] = llvm.call @air.convert.u.v2i8.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_30:.*]] = llvm.call @air.convert.s.v2i16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_31:.*]] = llvm.call @air.convert.u.v2i16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_32:.*]] = llvm.call @air.convert.f.v2f16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf16>
    // CHECK:  %[[CALL_33:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xbf16>
    // CHECK:  %[[CALL_34:.*]] = llvm.call @air.convert.f.v2f32.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf32>
    // CHECK:  %[[CALL_35:.*]] = llvm.call @air.convert.s.v2i8.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_36:.*]] = llvm.call @air.convert.u.v2i8.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_37:.*]] = llvm.call @air.convert.s.v2i16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_38:.*]] = llvm.call @air.convert.u.v2i16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_39:.*]] = llvm.call @air.convert.f.v2f16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf16>
    // CHECK:  %[[CALL_40:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xbf16>
    // CHECK:  %[[CALL_41:.*]] = llvm.call @air.convert.f.v2f32.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf32>
    // CHECK:  %[[CALL_42:.*]] = llvm.call @air.convert.s.v2i8.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_43:.*]] = llvm.call @air.convert.u.v2i8.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_44:.*]] = llvm.call @air.convert.s.v2i16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_45:.*]] = llvm.call @air.convert.u.v2i16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_46:.*]] = llvm.call @air.convert.s.v2i32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_47:.*]] = llvm.call @air.convert.u.v2i32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_48:.*]] = llvm.call @air.convert.f.v2bf16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_49:.*]] = llvm.call @air.convert.f.v2f32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xf32>
    // CHECK:  %[[CALL_50:.*]] = llvm.call @air.convert.s.v2i8.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_51:.*]] = llvm.call @air.convert.u.v2i8.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_52:.*]] = llvm.call @air.convert.s.v2i16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_53:.*]] = llvm.call @air.convert.u.v2i16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_54:.*]] = llvm.call @air.convert.s.v2i32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_55:.*]] = llvm.call @air.convert.u.v2i32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_56:.*]] = llvm.call @air.convert.f.v2f16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xf16>
    // CHECK:  %[[CALL_57:.*]] = llvm.call @air.convert.f.v2f32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xf32>
    // CHECK:  %[[CALL_58:.*]] = llvm.call @air.convert.s.v2i8.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi8>
    // CHECK:  %[[CALL_59:.*]] = llvm.call @air.convert.u.v2i8.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi8>
    // CHECK:  %[[CALL_60:.*]] = llvm.call @air.convert.s.v2i16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi16>
    // CHECK:  %[[CALL_61:.*]] = llvm.call @air.convert.u.v2i16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi16>
    // CHECK:  %[[CALL_62:.*]] = llvm.call @air.convert.s.v2i32.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi32>
    // CHECK:  %[[CALL_63:.*]] = llvm.call @air.convert.u.v2i32.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi32>
    // CHECK:  %[[CALL_64:.*]] = llvm.call @air.convert.f.v2f16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xf16>
    // CHECK:  %[[CALL_65:.*]] = llvm.call @air.convert.f.v2bf16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xbf16>
    %si8_to_si16 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, si16>

    %si8_to_ui16 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, ui16>

    %si8_to_si32 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, si32>

    %si8_to_ui32 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, ui32>

    %si8_to_f16 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, f16>

    %si8_to_bf16 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, bf16>

    %si8_to_f32 = pop.cast %si8_val : !pop.simd<2, si8> to !pop.simd<2, f32>

    %ui8_to_si16 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, si16>

    %ui8_to_ui16 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, ui16>

    %ui8_to_si32 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, si32>

    %ui8_to_ui32 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, ui32>

    %ui8_to_f16 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, f16>

    %ui8_to_bf16 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, bf16>

    %ui8_to_f32 = pop.cast %ui8_val : !pop.simd<2, ui8> to !pop.simd<2, f32>

    %si16_to_si8 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, si8>

    %si16_to_ui8 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, ui8>

    %si16_to_si32 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, si32>

    %si16_to_ui32 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, ui32>

    %si16_to_f16 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, f16>

    %si16_to_bf16 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, bf16>

    %si16_to_f32 = pop.cast %si16_val : !pop.simd<2, si16> to !pop.simd<2, f32>

    %ui16_to_si8 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, si8>

    %ui16_to_ui8 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, ui8>

    %ui16_to_si32 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, si32>

    %ui16_to_ui32 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, ui32>

    %ui16_to_f16 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, f16>

    %ui16_to_bf16 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, bf16>

    %ui16_to_f32 = pop.cast %ui16_val : !pop.simd<2, ui16> to !pop.simd<2, f32>

    %si32_to_si8 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, si8>

    %si32_to_ui8 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, ui8>

    %si32_to_si16 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, si16>

    %si32_to_ui16 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, ui16>

    %si32_to_f16 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, f16>

    %si32_to_bf16 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, bf16>

    %si32_to_f32 = pop.cast %si32_val : !pop.simd<2, si32> to !pop.simd<2, f32>

    %ui32_to_si8 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, si8>

    %ui32_to_ui8 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, ui8>

    %ui32_to_si16 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, si16>

    %ui32_to_ui16 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, ui16>

    %ui32_to_f16 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, f16>

    %ui32_to_bf16 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, bf16>

    %ui32_to_f32 = pop.cast %ui32_val : !pop.simd<2, ui32> to !pop.simd<2, f32>

    %f16_to_si8 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, si8>

    %f16_to_ui8 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, ui8>

    %f16_to_si16 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, si16>

    %f16_to_ui16 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, ui16>

    %f16_to_si32 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, si32>

    %f16_to_ui32 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, ui32>

    %f16_to_bf16 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, bf16>

    %f16_to_f32 = pop.cast %f16_val : !pop.simd<2, f16> to !pop.simd<2, f32>

    %bf16_to_si8 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, si8>

    %bf16_to_ui8 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, ui8>

    %bf16_to_si16 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, si16>

    %bf16_to_ui16 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, ui16>

    %bf16_to_si32 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, si32>

    %bf16_to_ui32 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, ui32>

    %bf16_to_f16 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, f16>

    %bf16_to_f32 = pop.cast %bf16_val : !pop.simd<2, bf16> to !pop.simd<2, f32>

    %f32_to_si8 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, si8>

    %f32_to_ui8 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, ui8>

    %f32_to_si16 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, si16>

    %f32_to_ui16 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, ui16>

    %f32_to_si32 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, si32>

    %f32_to_ui32 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, ui32>

    %f32_to_f16 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, f16>

    %f32_to_bf16 = pop.cast %f32_val : !pop.simd<2, f32> to !pop.simd<2, bf16>

    kgen.return
  }
}
