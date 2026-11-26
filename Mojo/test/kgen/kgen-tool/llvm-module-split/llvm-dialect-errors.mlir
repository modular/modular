// RUN: not llvm-module-split %s 2>&1 | FileCheck %s
// RUN: not llvm-module-split --per-func %s 2>&1 | FileCheck %s --check-prefix=PER-FUNC

// A constructor list names the symbols it runs, so it belongs with them
// rather than in every slice. The splitter rejects it.

// CHECK: error: 'llvm.mlir.global_ctors' op names a symbol, which is not supported when splitting a module per exported function
llvm.mlir.global_ctors ctors = [@init], priorities = [0 : i32], data = [#llvm.zero]

llvm.func @init() {
  llvm.return
}

llvm.func @f() {
  llvm.return
}

// The LLVM dialect splitter only splits per exported function.

// PER-FUNC: --per-func does not apply to an .mlir input
