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
}
