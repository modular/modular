//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerLIT
//===----------------------------------------------------------------------===//

def LowerLIT : Pass<"lower-lit", "mlir::ModuleOp"> {
  let summary = "Lower the `lit` dialects to the `kgen` dialect.";
  let description = [{
    This pass performs lowers high level abstractions from "lit" to "kgen"
    and diagnoses any errors that are found along the way.
  }];

  let dependentDialects = ["KGENDialect"];
}

//===----------------------------------------------------------------------===//
// LowerZAPToPOP
//===----------------------------------------------------------------------===//

def LowerZAPToPOP : Pass<"lower-zap-to-pop", "mlir::ModuleOp"> {
  let summary = "Lower the `zap` dialect to the `pop` dialect.";
  let description = [{
    This pass lowers `zap` dialect operations to `pop` dialect operations. This
    pass can run pre-elaboration.
  }];

  let dependentDialects = ["POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// LowerKGENToPOP
//===----------------------------------------------------------------------===//

def LowerKGENToPOP : Pass<"lower-kgen-to-pop", "mlir::ModuleOp"> {
  let summary = "Perform staged lowering of KGEN structs and lists.";
  let description = [{
    The `lower-kgen-to-pop` pass is a staged lowering of "special" KGEN types:
    structs and lists. Due to restrictions on how these compose with LLVM, we
    handle these lowerings as a separate pass to POP, a smaller set of
    operations to worry about than all of LLVM IR.
  }];

  let dependentDialects = ["POP::POPDialect", "mlir::index::IndexDialect"];
}

//===----------------------------------------------------------------------===//
// LowerKGENToLLVM
//===----------------------------------------------------------------------===//

def LowerKGENToLLVM : Pass<"lower-kgen-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `kgen` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion from the fully-elaborated and specified
    kgen.func ops and kgen types to the corresponding LLVM-dialect
    structures/types/ops.

    Set `c-call` to the names of top-level funcs for which to convert to
    C-compatible calling conventions. This breaks up argument and results
    structs in-place and converts arrays to be pass-by-reference. If any of the
    specified funcs have callsites in the module, the pass fails.

    For example, consider the following function that accepts and returns a
    buffer:

    ```mlir
    llvm.func @slice(%a: !llvm.struct<(i64, ptr<f32>)>, %i: !llvm.ptr<i64>)
        -> !llvm.struct<(i64, ptr<f32>)>
    ```

    The resulting signature will be:

    ```mlir
    llvm.func @slice(%a_0: i64, %a_1: !llvm.ptr<f32>, %i: !llvm.ptr<i64>,
                     %res_0: !llvm.ptr<i64>, %res_1: !llvm.ptr<ptr<f32>>)
    ```
  }];

  // We generate the LLVM dialect.
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    ListOption<"topLevelFuncs", "c-call", "std::string",
               "top-level funcs to make callable from C">
  ];
}

//===----------------------------------------------------------------------===//
// LowerPOPToLLVM
//===----------------------------------------------------------------------===//

def LowerPOPToLLVM : Pass<"lower-pop-to-llvm"> {
  let summary = "Lower the `pop` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion of fully-elaborated and specified `pop`
    dialect operations to LLVM dialect operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// LowerGlobalPOPToLLVM
//===----------------------------------------------------------------------===//

def LowerGlobalPOPToLLVM : Pass<"lower-global-pop-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower global `pop` dialect operations to the `llvm` dialect.";
  let description = [{
    This module pass lowers `pop` operations that have global implications,
    and thus their lowering pass needs to run on the whole module. These are
    external calls and glboal constant operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// LowerSCFToLLVM
//===----------------------------------------------------------------------===//

def LowerSCFToLLVM : Pass<"lower-scf-to-llvm"> {
  let summary = "Lower the `scf` dialect to the `llvm` dialect.";
  let description = [{
    This pass converts structured control-flow operations to LLVM operations and
    control-flow graph regions. Because this pass generates CFG regions, it must
    run after `lower-kgen-to-llvm`, which lowers `kgen.func` to `llvm.func`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// ElaborateGenerators
//===----------------------------------------------------------------------===//

def ElaborateGenerators : Pass<"elaborate-generators", "mlir::ModuleOp"> {
  let summary = "elaborate generators into fully concrete funcs";
  let description = [{
    This pass performs the elaboration step for the kgen pipeline,
    converting generators and generator interfaces into concrete
    funcs.
  }];

  let dependentDialects = ["::M::KGEN::KGENDialect"];

  let options = [
    ListOption<"searchPaths", "search-path", "std::string",
               "Search path for imported modules.">,
    Option<"shouldDoSearch", "enable-search", "bool", "false",
           "If true, benchmark implementations to find the fastest one.">,
  ];
}

//===----------------------------------------------------------------------===//
// Resolve Includes
//===----------------------------------------------------------------------===//

def ResolveIncludes : Pass<"resolve-includes", "mlir::ModuleOp"> {
  let summary = "resolve includes in the module";
  let description = [{
    This pass performs include resolution on the KGEN module, replacing all
    `kgen.include` operations with the included modules and reconciling included
    symbols.
  }];

  let options = [
    ListOption<"searchPaths", "search-path", "std::string",
               "Search path for imported modules.">,
  ];
}

//===----------------------------------------------------------------------===//
// EmitLLVM
//===----------------------------------------------------------------------===//

def EmitLLVM : Pass<"emit-llvm", "mlir::ModuleOp"> {
  let summary = "emit LLVM IR";
  let description = [{
    This pass emits the LLVM IR for the concrete KGEN. If no file is provided
    then the IR will be printed to stdout.
  }];

  let dependentDialects = ["::M::KGEN::KGENDialect"];

  let options = [
    Option<"output", "output", "std::string", "", "Output file path.">,
  ];
}

//===----------------------------------------------------------------------===//
// PruneImpossibleVariants
//===----------------------------------------------------------------------===//

def PruneImpossibleVariants: Pass<"prune-impossible-variants",
                                  "mlir::ModuleOp"> {
  let summary = "remove variant cases that are known to be impossible";
  let description = [{
    The `prune-impossible-variants` pass runs a simple forward data-flow
    analysis to determine variant cases that are impossible.

    The pass folds `pop.variant.is` operations to constants when the operand is
    known to be or not to be the queried type and rewrites function signatures
    that contain variants of known types to those types. This pass is meant to
    be run pre-elaboration.
  }];
}
