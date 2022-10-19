//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the LIT dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/LITDialect/LITDialect.td"
include "KGEN/KGENDialect/KGENDeclInterface.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/POPDialect/POPTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for the operation in this dialect.
class LITOp<string mnemonic, list<Trait> traits = []> :
    Op<LITDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LITFuncOp
//===----------------------------------------------------------------------===//

def LITFuncOp
  : LITOp<"func", [FunctionOpInterface, Symbol, IsolatedFromAbove,
                   SingleBlockImplicitTerminator<"KGEN::ReturnOp">,
                   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                   DeclareOpInterfaceMethods<KGENDeclInterface>,
                   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "High level generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       KStringArrayAttr:$valueParamNames,
                       TypeAttrOf<FunctionType>:$function_type,
                       LinkageAttr:$linkage,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<LITFuncOp>::getBody;
    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LITStructDeclOp
//===----------------------------------------------------------------------===//

def LITStructDeclOp : LITOp<"struct.decl", [Symbol,
                  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                  DeclareOpInterfaceMethods<KGENDeclInterface>, NoTerminator]> {
  let summary = "Struct declaration.";
  let description = [{
    The `lit.struct.decl` operation represents a Lightning struct declaration.
    This models the syntactic structure, which can contain functions and other
    things just like Lightning source.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, ParamDeclArrayAttr:$paramDecls,
                       // Needed for KGENDeclInterface. Always empty.
                       KTypeArrayAttr:$resultParamTypes);
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name `` custom<OptionalParameterSpec>($paramDecls, $resultParamTypes)
    attr-dict-with-keyword
    $fields
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

def VarDeclOp : LITOp<"var.decl",
         [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Local variable declaration for Lightning";

  let arguments = (ins StrAttr:$name);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}
