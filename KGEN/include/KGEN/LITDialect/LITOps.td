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
                   NoTerminator,
                   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                   DeclareOpInterfaceMethods<KGENDeclInterface>,
                   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let summary = "High level generator definition";

let description = [{
    The `lit.func` operation models a def/fn declaration at the Lit source
    level.
    @interface and @staticmethod decorators in Lit correspond to the
    attributes `isInterface` and `isStatic` respectively.
    When `isInterface` is set, `lit.func` has an empty body block.
    The optional `implements` attribute refers to the interface name this
    `lit.func` implement and cannot be set alongside `isInterface`.
  }];

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       KStringArrayAttr:$valueParamNames,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isStatic,
                       UnitAttr:$isInterface,
                       UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
                       UnitAttr:$raises,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    Block *getBody() { return &getFunctionBody().back(); }
    ReturnOp getReturnOp();

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the result type, which is guaranteed to be present.
    Type getResultType() { return getFunctionType().getResults()[0]; }
  }];

  let builders = [
    // Create a func with a region and a body block.
    OpBuilder<(ins "StringAttr":$name)>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LITRaiseErrorOp
//===----------------------------------------------------------------------===//

def LITRaiseErrorOp : LITOp<"raise_error", [Pure]> {
  let summary = "Form an error value.";
  let description = [{
    The `lit.raise_error` operation wraps a `Error` (library type) in a
    `!lit.raises_or` value that can be returned from a method that is marked as
    raising.

    Example:

    ```mlir
    %result = lit.raise_error %err : <@Error> -> <!kgen.declref<@Int>>
    ```
  }];

  let arguments = (ins DeclRefType:$error);
  let results = (outs LITRaisesOrType:$result);

  let assemblyFormat = "$error attr-dict `:` type($error) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// LITFormValueOp
//===----------------------------------------------------------------------===//

def LITFormValueOp : LITOp<"form_value", [Pure,
    TypesMatchWith<"value type", "result", "value",
                   "cast<RaisesOrType>($_self).getType()">]> {
  let summary = "Wrap a valid method result.";
  let description = [{
    The `lit.form_value` operation takes a valid method result value and wraps
    it in a `!lit.raises_or` value that can be returned from a method that is
    marked as raising.

    Example:

    ```mlir
    %result = lit.form_value %value : <!kgen.declref<@Int>>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs LITRaisesOrType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
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
