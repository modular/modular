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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<LITFuncOp>::getBody;
    ReturnOp getReturnOp();

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the result type, which is guaranteed to be present.
    Type getResultType() { return getFunctionType().getResults()[0]; }

    /// If this is a special function like __init__ return the enum that
    /// identifies it, otherwise return kNormal.
    SpecialFunctionKind getSpecialFunctionKind();
  }];

  let builders = [
    // Create a func with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name, "FlatSymbolRefAttr":$implements)>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LITStructDeclOp
//===----------------------------------------------------------------------===//

def LITStructDeclOp : LITOp<"struct.decl", [Symbol, SymbolTable,
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

  let extraClassDeclaration = [{
    /// Get the field declarations.
    auto getFieldDecls() { return getFields().getOps<VarDeclOp>(); }
    auto field_begin() { return getFieldDecls().begin(); }
    auto field_end() { return getFieldDecls().end(); }
  }];

  let builders = [
    // Create a struct with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name)>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LITStructGEPOp
//===----------------------------------------------------------------------===//

def LITStructGEPOp : LITOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a pointer to a struct field.";
  let description = [{
    The `lit.struct.gep` operation computes a pointer to a struct field by
    name from a parametric struct.
  }];

  let arguments = (ins StrAttr:$field, PointerOf<RefType>:$container);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($result) `from` type($container)
  }];
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
