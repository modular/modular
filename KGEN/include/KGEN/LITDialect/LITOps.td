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
include "KGEN/LITDialect/LITTypes.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENDeclInterface.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/HLCFDialect/HLCFInterfaces.td"
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
                       TypeAttrOf<SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isStatic,
                       UnitAttr:$isInterface,
                       UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
                       UnitAttr:$raises,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    using KGENDeclInterfaceTrait::getFunctionType;
    using KGENDeclInterfaceTrait::getFunctionTypeAttr;

    Block *getBody() { return &getFunctionBody().back(); }
    ReturnOp getReturnOp();

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
// LITUnwrapOrPropagateOp
//===----------------------------------------------------------------------===//

def LITUnwrapOrPropagateOp : LITOp<"unwrap_or_propagate", [
    TypesMatchWith<"value type", "value", "result",
                   "cast<RaisesOrType>($_self).getType()">]> {
  let summary = "Unwrap a value or propagate an error.";
  let description = [{
    The `lit.unwrap_or_propagate` operation takes a `!lit.raises_or` value and
    returns the unwrapped value if it is not an error and propagates the error
    if it is one.

    Example:

    ```mlir
    %value = lit.unwrap_or_propagate %maybeErr : <!kgen.declref<@Int>>
    ```
  }];

  let arguments = (ins LITRaisesOrType:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// LITTryOp
//===----------------------------------------------------------------------===//

def LITTryOp : LITOp<"try", [RecursivelySpeculatable, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<ControlFlowNode>]> {
  let summary = "A try-except-else block.";
  let description = [{
    The `lit.try` operation represents a try-except-else statement. When inside
    the 'try' region, any implicitly propagated errors interrupt execution and
    branch to the 'except' region. If the 'try' region completes execution
    without any exceptions, control-flow branches to the 'else' region.

    Example:

    ```mlir
    lit.try {
      %0 = kgen.call @maybeRaises() : () -> !lit.raises_or<!kgen.declref<@Int>>
      %1 = lit.unwrap_or_propagate %0 : <!kgen.declref<@Int>>
      lit.try.yield
    } except (%err: !kgen.declref<@Error>) {
      lit.try.yield
    } else {
      lit.try.yield
    }
    ```
  }];

  let regions = (region MinSizedRegion<1>:$tryRegion,
                        MinSizedRegion<1>:$exceptRegion,
                        MinSizedRegion<1>:$elseRegion);
  let assemblyFormat = [{
    $tryRegion `except` custom<ExceptRegion>($exceptRegion) `else` $elseRegion
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LITTryYieldOp
//===----------------------------------------------------------------------===//

def LITTryYieldOp : LITOp<"try.yield", [Pure, Terminator,
    HasParent<"LITTryOp">, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block terminator.";
  let description = [{
    The `lit.try.yield` operation is used as a terminator to `lit.try` regions.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LITTryRaiseOp
//===----------------------------------------------------------------------===//

def LITTryRaiseOp : LITOp<"try.raise", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block raise an error.";
  let description = [{
    The `lit.try.raise` operation is used within a `lit.try` try region to
    propagate an error to the except region.
  }];

  let arguments = (ins AnyTypeOf<[DeclRefType, POP_StructType]>:$error);
  let assemblyFormat = "$error attr-dict `:` type($error)";
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
