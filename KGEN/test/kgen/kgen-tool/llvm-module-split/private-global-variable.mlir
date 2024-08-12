// RUN: kgen-translate %s --mlir-to-llvmir -o %t
// RUN: llvm-module-split %t --per-func | FileCheck %s

llvm.mlir.global internal constant @str0("str0\00") {addr_space = 0 : i32, alignment = 16 : i64}
llvm.mlir.global private constant @str1("str1\00") {addr_space = 0 : i32, alignment = 16 : i64}

llvm.func @f(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.addressof @str0: !llvm.ptr
  %1 = llvm.mlir.addressof @str1: !llvm.ptr
  llvm.call @f(%0) : (!llvm.ptr) -> ()
  llvm.call @f(%1) : (!llvm.ptr) -> ()
  llvm.call @f(%arg0) : (!llvm.ptr) -> ()
  llvm.return
}

llvm.func @h() {
  %0 = llvm.mlir.addressof @str0: !llvm.ptr
  %1 = llvm.mlir.addressof @str1: !llvm.ptr
  llvm.call @f(%0) : (!llvm.ptr) -> ()
  llvm.call @f(%1) : (!llvm.ptr) -> ()
  llvm.return
}

// COM: check private global variable is handled 
// CHECK: [LLVM Module Split: submodule 0]
// CHECK: @str0 = weak constant [5 x i8] c"str0\00", align 16
// CHECK: @str1 = weak constant [5 x i8] c"str1\00", align 16
// CHECK: define void @f(ptr %0) 
// CHECK: [LLVM Module Split: submodule 1]
// CHECK: @str0 = weak constant [5 x i8] c"str0\00", align 16
// CHECK: @str1 = weak constant [5 x i8] c"str1\00", align 16
// CHECK: declare void @f(ptr)
// CHECK: define void @h() 
