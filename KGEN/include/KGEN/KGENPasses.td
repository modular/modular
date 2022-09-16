//===- KGENPasses.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerHLKGEN
//===----------------------------------------------------------------------===//

def LowerHLKGEN : Pass<"lower-hlkgen", "mlir::ModuleOp"> {
  let summary = "Lower the `hlkgen` dialects to the `kgen` dialect.";
  let description = [{
    This pass performs lowers high level abstractions from "hlkgen" to "kgen"
    and diagnoses any errors that are found along the way.
  }];

  let dependentDialects = ["KGENDialect"];

  let constructor = "::M::KGEN::createLowerHLKGENPass()";
}

//===----------------------------------------------------------------------===//
// ConvertZAPToPOP
//===----------------------------------------------------------------------===//

def LowerZAPToPOP : Pass<"lower-zap-to-pop"> {
  let summary = "Lower the `zap` dialects to the `pop` and `meta` dialects.";
  let description = [{
    This pass lowers `zap` dialect operations to combinations of `pop` and
    `meta` dialect operations. This pass can run pre-elaboration.
  }];

  let dependentDialects = ["MetaDialect", "POP::POPDialect"];

  let constructor = "createLowerZAPToPOPPass()";
}

//===----------------------------------------------------------------------===//
// ConvertKGENToLLVM
//===----------------------------------------------------------------------===//

def ConvertKGENToLLVM : Pass<"convert-kgen-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `kgen` and `meta` dialects to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion from the fully-elaborated and specified
    kgen.func ops and kgen/meta types to the corresponding LLVM-dialect
    structures/types/ops.

    Set `break-up-structs` to the names of top-level funcs for which to break
    up argument and results structs in-place. If any of the specified funcs
    have callsites in the module, the pass fails.

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

    Set `emit-c-wrappers` to the names of any funcs for which to emit C
    wrappers. C wrappers contain unpacked struct arguments and results. They
    re-pack the struct and invoke the original func. For example:

    ```mlir
    llvm.func @slice(%a: !llvm.struct<(i64, ptr<f32>)>, %i: !llvm.ptr<i64>)
        -> !llvm.struct<(i64, ptr<f32>)>
      attributes {c_wrapper = @slice_c_wrapper}

    llvm.func @slice_c_wrapper(
        %a_0: i64, %a_1: !llvm.ptr<f32>, %i: !llvm.ptr<i64>,
        %res_0: !llvm.ptr<i64>, %res_1: !llvm.ptr<ptr<f32>>)
    ```

    The original function is tagged with a reference to the wrapper. If a
    function is listed under both `break-up-structs` and `emit-c-wrappers`, a
    wrapper is still generated for the function, but since the original function
    will have no structs, the wrapper will be a forwarding call.

    If `emit-opaque-wrappers` is set, top-level funcs listed under
    `break-up-structs` will get an additional opaque wrapper in the form:

    ```
    llvm.func @slice_opaque_wrapper(
        %a: !llvm.ptr<struct<packed (i64, ptr, i8)>>, %i: !llvm.ptr<i64>,
        %res: !llvm.ptr<struct<packed (i64, ptr, i8)>>)
    ```

    Where each struct argument in the original function is replaced with a
    pointer to a packed struct (1-byte alignment, no padding) of the same
    element types. If the result was a struct, then a pointer to a packed result
    struct is appended to the argument list.

    The opaque wrapper can be used to generically invoke funcs with instances
    of custom types if those custom types have some way to generically allocate
    and populate themselves into a packed struct.
  }];

  // We generate the LLVM dialect.
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "::M::KGEN::createConvertKGENToLLVMPass()";

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    ListOption<"breakUpStructs", "break-up-structs", "std::string",
               "top-level funcs to make callable from C">,
    Option<"emitOpaqueWrappers", "emit-opaque-wrappers", "bool",
           /*default=*/"false", "whether to emit opaque wrappers">,
    ListOption<"emitCWrappers", "emit-c-wrappers", "std::string",
               "funcs for which to emit C wrappers">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertPOPToLLVM
//===----------------------------------------------------------------------===//

def ConvertPOPToLLVM : Pass<"convert-pop-to-llvm"> {
  let summary = "Lower the `pop` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion of fully-elaborated and specified `pop`
    dialect operations to LLVM dialect operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "::M::KGEN::createConvertPOPToLLVMPass()";

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
  let constructor = "::M::KGEN::createLowerGlobalPOPToLLVM()";

  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// ConvertSCFToLLVM
//===----------------------------------------------------------------------===//

def ConvertSCFToLLVM : Pass<"convert-scf-to-llvm"> {
  let summary = "Lower the `scf` dialect to the `llvm` dialect.";
  let description = [{
    This pass converts structured control-flow operations to LLVM operations and
    control-flow graph regions. Because this pass generates CFG regions, it must
    run after `convert-kgen-to-llvm`, which lowers `kgen.func` to `llvm.func`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "::M::KGEN::createConvertSCFToLLVMPass()";

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
  let constructor = "::M::KGEN::createElaborateGeneratorsPass()";

  let options = [
    ListOption<"searchPaths", "search-path", "std::string",
               "Search path for imported modules.">,
  ];
}
