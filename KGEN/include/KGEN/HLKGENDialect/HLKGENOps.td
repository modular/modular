//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the HLKGEN dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "KGEN/HLKGENDialect/HLKGENDialect.td"
include "KGEN/KGENDialect/KGENDeclInterface.td"
include "KGEN/POPDialect/POPTypes.td"

// Base class for the operation in this dialect.
class HLKGENOp<string mnemonic, list<Trait> traits = []> :
    Op<HLKGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// HLGeneratorOp
//===----------------------------------------------------------------------===//

def HLGeneratorOp
  : HLKGENOp<"generator", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                           Symbol, IsolatedFromAbove,
                           SingleBlockImplicitTerminator<"KGEN::ReturnOp">,
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                           DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "High level generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$sym_visibility,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<HLGeneratorOp>::getBody;
    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

def VarDeclOp : HLKGENOp<"var.decl", []> {
  let summary = "Local variable declaration for Lightning";

  let arguments = (ins StrAttr:$name);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$name `` `:` type($result) attr-dict";
}
