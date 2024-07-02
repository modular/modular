// RUN: kgen-opt %s -split-input-file -lower-kgen-to-llvm | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @kernel
  // CHECK-SAME: %[[SIZE:.*]]: [[INDEXTY:.*]], %[[PTR:.*]]: !llvm.ptr, %[[DTYPE:.*]]: i8
  // CHECK-SAME: %[[SIZE_OUT:.*]]: !llvm.ptr, %[[PTR_OUT:.*]]: !llvm.ptr, %[[DTYPE_OUT:.*]]: !llvm.ptr
  // CHECK: %[[BUFFER:.*]] = llvm.mlir.undef
  // CHECK: %[[B0:.*]] = llvm.insertvalue %[[SIZE]], %[[BUFFER]][0]
  // CHECK: %[[B1:.*]] = llvm.insertvalue %[[PTR]], %[[B0]][1]
  // CHECK: %[[RESULT_BUFFER:.*]] = llvm.insertvalue %[[DTYPE]], %[[B1]][2]
  // CHECK: %[[SIZE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][0]
  // CHECK: llvm.store %[[SIZE_RESULT]], %[[SIZE_OUT]]
  // CHECK: %[[PTR_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][1]
  // CHECK: llvm.store %[[PTR_RESULT]], %[[PTR_OUT]]
  // CHECK: %[[DTYPE_RESULT:.*]] = llvm.extractvalue %[[RESULT_BUFFER]][2]
  // CHECK: llvm.store %[[DTYPE_RESULT]], %[[DTYPE_OUT]]
  // CHECK: llvm.return

  kgen.func export C @kernel(%a: !kgen.struct<(index, pointer<simd<1, invalid>>, dtype)>) -> !kgen.struct<(index, pointer<simd<1, invalid>>, dtype)> {
    kgen.return %a : !kgen.struct<(index, pointer<simd<1, invalid>>, !kgen.dtype)>
  }

}

// -----

!nestedStruct = !kgen.struct<(struct<()>, struct<(f32)>)>

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @kernel
  // CHECK-SAME: %[[V:.*]]: f32 {llvm.noundef}, %[[V_OUT:.*]]: !llvm.ptr
  // CHECK-NEXT: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(f32)>)>
  // CHECK-NEXT: %[[EMPTY:.*]] = llvm.mlir.undef : !llvm.struct<()>
  // CHECK-NEXT: %[[S1:.*]] = llvm.insertvalue %[[EMPTY]], %[[S0]][0]
  // CHECK-NEXT: %[[S2:.*]] = llvm.mlir.undef : !llvm.struct<(f32)>
  // CHECK-NEXT: %[[S3:.*]] = llvm.insertvalue %[[V]], %[[S2]][0]
  // CHECK: %[[ARG:.*]] = llvm.insertvalue %[[S3]], %[[S1]][1]
  // CHECK: %[[S4:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: %[[V_RESULT:.*]] = llvm.extractvalue %[[S4]][0]
  // CHECK: llvm.store %[[V_RESULT]], %[[V_OUT]]

  kgen.func export C @kernel(%a: !nestedStruct) -> !nestedStruct {
    kgen.return %a : !nestedStruct
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @kernel
  // CHECK-SAME: %[[I:.*]]: f32
  // CHECK: llvm.return

  kgen.func export C @kernel(%i: f32, %a: !kgen.struct<(index, pointer<simd<1, f32>>, dtype)>) {
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @kernel
  // CHECK-SAME: -> i64
  // CHECK: %[[RESULT:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.return %[[RESULT]]

  kgen.func export C @kernel(%a: !kgen.struct<(index, pointer<simd<1, f32>>, dtype)>) -> i64 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    kgen.return %0 : i64
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK-LABEL: @kernel
  // CHECK-SAME: (%[[ARG:.*]]: f32 {llvm.noundef}) -> f32
  // CHECK-NEXT: return %[[ARG]] : f32
  kgen.func export C @kernel(%a: f32) -> f32 {
    kgen.return %a : f32
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func export C @foo() {
    kgen.return
  }

  // CHECK-LABEL: llvm.func @foo
  // CHECK-LABEL: llvm.func @call_foo

  kgen.func export C @call_foo() {
    kgen.call @foo() : () -> ()
    kgen.return
  }
}
