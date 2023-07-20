//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITDECLINTERFACE_TD
#define KGEN_LITDIALECT_LITDECLINTERFACE_TD

include "mlir/IR/OpBase.td"

def ASTDeclInterface : OpInterface<"ASTDeclInterface"> {
  let description = [{
    This interface represents an operation that corresponds to an ASTDecl,
    generally corresponding to a declaration in the Mojo source language. For
    example, functions, variables, structs, etc.
  }];
  let cppNamespace = "::M::KGEN::LIT";

  let methods = [
    InterfaceMethod<[{
      Return the name of this decl.
    }], "StringAttr", "getDeclName", (ins), [{}], [{
      return $_op.getNameAttr();
    }]>,

    InterfaceMethod<[{
      Return the documentation string attached to this decl, or nullptr if the
      decl has no documentation attached.
    }], "DocStringAttr", "getDocStringAttr", (ins), [{}], [{
      return DocStringAttr();
    }]>,

    InterfaceMethod<[{
      Set the documentation string for this decl.
    }], "void", "setDocStringAttr", (ins "DocStringAttr":$docString), [{}], [{
      llvm_unreachable("TODO: decl does not support a doc string");
    }]>,
  ];
}

#endif // KGEN_LITDIALECT_LITDECLINTERFACE_TD
