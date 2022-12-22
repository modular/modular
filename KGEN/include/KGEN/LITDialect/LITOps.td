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
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/HLCFDialect/HLCFInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class LITOp<string mnemonic, list<Trait> traits = []> :
    Op<LITDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LITFuncOp
//===----------------------------------------------------------------------===//

def LIT_FuncOp : LITOp<"func", [
    FunctionOpInterface, Symbol, IsolatedFromAbove, NoTerminator,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclInterface, FuncInterface]> {
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
                       KGEN_StringArrayAtrr:$valueParamNames,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isStatic,
                       UnitAttr:$isInterface,
                       UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
                       UnitAttr:$raises,
                       OptionalAttr<SymbolRefAttr>:$implements);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    Block *getBody() { return &getFunctionBody().back(); }
    ReturnOp getReturnOp();

    /// Returns the result type, which is guaranteed to be present.
    Type getResultType() { return getFunctionType().getResults()[0]; }

    /// Return the normal result type.  This is the same as getResultType unless
    /// the function throws, in which case this is dug out of the variant.
    Type getNormalResultType();

    /// Return the SymbolRefAttr for a declaration, including all scoping that
    /// may be needed, making it unique for every declaration.
    SymbolRefAttr getSymbolRef();

    /// Return a SymbolConstantAttr for this function, optionally bound to a set
    /// of parameter bindings.
    SymbolConstantAttr getBoundReference(ParamBindArrayAttr bindings = {});
  }];

  let builders = [
    // Create a func with a region and a body block.
    OpBuilder<(ins)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def FileModuleOp : LITOp<"file_module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol
  ] # GraphRegionNoTerminator.traits> {
  let summary = "source file module";
  let description = [{
    A `lit.file_module` models a file module at the lit source level. It
    contains any functions, modules, variables, etc. defined within the module.

    Example:

    ```mlir
    lit.file_module @bar {
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>,
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UnwrapOrPropagateOp
//===----------------------------------------------------------------------===//

def LIT_OrVariant : ConfinedType<POP_VariantType,
    [CPred<"cast<POP::VariantType>($_self).getTypes().size() == 2">],
    "variant with 2 subtypes", "POP::VariantType">;

def LIT_UnwrapOrPropagateOp : LITOp<"unwrap_or_propagate", [
    TypesMatchWith<"value type", "value", "result",
                   "cast<POP::VariantType>($_self).getType(1)">]> {
  let summary = "Unwrap a value or propagate an error.";
  let description = [{
    The `lit.unwrap_or_propagate` operation takes a variant of error and value
    and returns the unwrapped value if it is not an error and propagates the
    error if it is one.

    Example:

    ```mlir
    %value = lit.unwrap_or_propagate %maybeErr : <@Error, @Int>
    ```
  }];

  let arguments = (ins LIT_OrVariant:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def LIT_TryOp : LITOp<"try", [RecursivelySpeculatable, RecursiveMemoryEffects,
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
      %0 = kgen.call @maybeRaises() : () -> !pop.variant<@Error, @Int>
      %1 = lit.unwrap_or_propagate %0 : <@Error, @Int>
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
// TryYieldOp
//===----------------------------------------------------------------------===//

def LIT_TryYieldOp : LITOp<"try.yield", [Pure, Terminator,
    HasParent<"TryOp">, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block terminator.";
  let description = [{
    The `lit.try.yield` operation is used as a terminator to `lit.try` regions.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

def LIT_TryRaiseOp : LITOp<"try.raise", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block raise an error.";
  let description = [{
    The `lit.try.raise` operation is used within a `lit.try` try region to
    propagate an error to the except region.
  }];

  let arguments = (ins AnyTypeOf<[KGEN_DeclRefType, POP_StructType]>:$error);
  let assemblyFormat = "$error attr-dict `:` type($error)";
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

def LIT_VarDeclOp : LITOp<"var.decl",
         [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Local variable declaration for Lightning";

  let arguments = (ins StrAttr:$name);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}
