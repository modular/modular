// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @external_call
  kgen.func @external_call(%a: !pop.simd<1, ui32>, %b: !pop.pointer<i32>) -> !pop.simd<4, f64> {
    // CHECK: llvm.call @foo
    %0 = pop.external_call @foo(%a) attributes {
      funcAttrs = ["noinline", "noreturn"],
      memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>
    } : (!pop.simd<1, ui32>) -> !pop.simd<4, f64>
    // CHECK: llvm.call @bar
    %1 = pop.external_call @bar(%b) attributes {argAttrs = [{llvm.noalias}], resAttrs = [{llvm.signext}]} : (!pop.pointer<i32>) -> i32
    kgen.return %0 : !pop.simd<4, f64>
  }
  // CHECK: llvm.func @foo(i32) -> vector<4xf64>
  // CHECK-SAME: memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>
  // CHECK-SAME: passthrough = ["noinline", "noreturn"
  // CHECK: llvm.func @bar(!llvm.ptr<i32> {llvm.noalias}) -> (i32 {llvm.signext})
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
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

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @global_constant
  kgen.func @global_constant() {
    // CHECK: llvm.mlir.addressof @global_constant_0
    %0 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_0
    %1 = pop.global_constant: ui32 = <5>
    // CHECK: llvm.mlir.addressof @global_constant_1
    %2 = pop.global_constant: simd<2, si32> = <<2, 5>>
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
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

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
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

// COM: Don't generate globals where there are none.
module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-NOT: llvm.mlir.global_ctors
  // CHECK-NOT: llvm.mlir.global_dtors
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK: llvm.mlir.global_ctors {ctors = [@foo_c, @bar_c, @noop], priorities = [2 : i32, 5 : i32, 0 : i32]}
  // CHECK: llvm.mlir.global_dtors {dtors = [@foo_d, @bar_d, @noop], priorities = [2 : i32, 5 : i32, 0 : i32]}
  llvm.func @foo_c() {
    llvm.return
  }
  llvm.func @foo_d() {
    llvm.return
  }
  llvm.func @bar_c() {
    llvm.return
  }
  llvm.func @bar_d() {
    llvm.return
  }

  // CHECK: llvm.mlir.global internal @foo() {{.*}} : i32
  kgen.global @foo : i32 (2, @foo_c, @foo_d)
  // CHECK: llvm.mlir.global internal @bar() {{.*}} : i64
  kgen.global @bar : i64 (5, @bar_c, @bar_d)

  llvm.func @noop() {
    llvm.return
  }
  // CHECK: llvm.mlir.global external @exported() {{.*}} : f32
  // CHECK-NEXT: [[UNDEF:%.*]] = llvm.mlir.undef
  // CHECK-NEXT: llvm.return [[UNDEF]]
  kgen.global export @exported : f32 (0, @noop, @noop)
}

// -----

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: llvm.func @alloc_free
  llvm.func @alloc_free() {
    %size = index.constant 1
    %align = index.constant 8
    // CHECK: [[RAW_PTR:%.*]] = llvm.call @kgenAlignedAlloc
    // CHECK-NEXT: [[PTR:%.*]] = llvm.bitcast [[RAW_PTR]] : !llvm.ptr to !llvm.ptr<i64>
    %0 = pop.aligned_alloc %align, %size : <index>
    // CHECK: [[RAW_PTR:%.*]] = llvm.bitcast [[PTR]] : !llvm.ptr<i64> to !llvm.ptr
    // CHECK-NEXT: llvm.call @kgenAlignedFree([[RAW_PTR]])
    pop.aligned_free %0 : <index>
    llvm.return
  }

  // CHECK: llvm.func @kgenAlignedAlloc(i64 {llvm.allocalign}, i64) -> (!llvm.ptr {llvm.noalias})
  // CHECK-DAG: ["allockind", "41"]
  // CHECK-DAG: ["allocsize", "8589934591"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]

  // CHECK: llvm.func @kgenAlignedFree(!llvm.ptr {llvm.allocptr})
  // CHECK-DAG: ["allockind", "4"]
  // CHECK-DAG: ["alloc-family", "kgen_aligned_allocator"]
}
