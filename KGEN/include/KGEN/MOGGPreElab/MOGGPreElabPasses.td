//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def AnnotateKernels : Pass<"mogg-annotate-kernels", "ModuleOp"> {
  let summary = "Strips MOGG decorators and replaces them with attributes";
  let description = [{
    A MOGG domain specific pass which will annotate each kernel at the lit
    level allowing subsequent passes to use the information parsed in a
    sanitized way.
  }];
}

def VerifyKernels : Pass<"mogg-verify-kernels", "ModuleOp"> {
  let summary = "Verifies that kernels are well constructed";
  let description = [{
    This pass ensures that the kernels present in the module are well
    constructed. Among other things, it checks that shape functions match
    execute functions and that inputs appear after outputs in the function's
    signature.
  }];
}

def DumpKernels : Pass<"mogg-dump-kernels", "ModuleOp"> {
  let summary = "Prints the internal representation of kernels";
  let description = [{
    Prints the internal in-memory representation of kernels for debugging
    purposes. This pass can be run on both `LIT` and `KGEN` kernels because the
    underlying representation is compatible with both.
  }];
}
