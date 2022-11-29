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
// LITStructCreateOp
//===----------------------------------------------------------------------===//

def LITStructCreateOp : LITOp<"struct.create", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Create a parametric struct.";
  let description = [{
    The `lit.struct.create` operation creates a struct of the referred type
    with the provided values as fields. The operands must match the fields in
    the order they are declared.

    Example:

    ```mlir
    kgen.struct.decl @MyStruct<T: dtype> {
      a : !pop.scalar<T>
      b : !pop.scalar<f32>
      c : !pop.simd<2, T>
    }

    %0 = kgen.struct.create(%a, %b, %c) :
      (!pop.scalar<type>, !pop.scalar<f32>, !pop.simd<2, type>)
      -> !kgen.declref<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs DeclRefType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// LITStructExtractOp
//===----------------------------------------------------------------------===//

def LITStructExtractOp : LITOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Extract an element from a struct field.";
  let description = [{
    The `lit.struct.extract` operation extracts a field from a struct by name.
  }];

  let arguments = (ins StrAttr:$field, DeclRefType:$structVal);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $structVal `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($result) `from` type($structVal)
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBase, "VarDeclOp":$field)>
  ];
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

  let arguments = (ins StrAttr:$field, PointerOf<DeclRefType>:$container);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($result) `from` type($container)
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBasePtr, "VarDeclOp":$field)>
  ];
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
