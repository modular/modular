//===- HLKGENOps.td -------------------------------------------------------===//
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
include "KGEN/KGENDialect/KGENDeclInterface.td"

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
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       ParamDeclArrayAttr:$resultParamDecls,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getBody().front();
    }

    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}
