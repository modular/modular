//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
#define KGEN_KGENDIALECT_KGENDECLINTERFACE_TD

include "mlir/IR/OpBase.td"

def KGENDeclInterface : OpInterface<"KGENDeclInterface"> {
  let description = [{
    This is an interface common to kgen.func, kgen.generator,
    kgen.generator.interface, as well as lit.func.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the KGEN declaration scope is parametrically isolated from
      above, meaning that operations at and below the scope cannot reference
      parameters from higher scopes, if any.

      Note that 'false' does not mean the scope has references to parameters in
      higher scopes but that it *can* have such references.
    }], "bool", "isIsolatedFromAbove", (ins), [{}], [{ return false; }]>,
    InterfaceMethod<[{
      Notify a KGEN declaration scope that allows references to parameters in
      higher scopes that it has no such references in its body.
    }], "void", "notifyKnownIsolatedFromAbove", (ins), [{}], [{ }]>,

    InterfaceMethod<[{
      Return the parameter + SSA value signature.
    }], "SignatureType", "getSignature", (ins), [{}], [{
      return SignatureType::get($_op.getParamDeclsAttr(),
                                $_op.getResultParamTypesAttr(),
                                $_op.getFunctionType());
    }]>,

    InterfaceMethod<[{
      Return the parameter + SSA value signature including contextual parameters.
    }], "SignatureType", "getFullSignature", (ins), [{}], [{
      return KGEN::getFullSignature($_op);
    }]>,

    // Normal arguments and results.
    InterfaceMethod<[{
      Return the FunctionType for input/result SSA values.
    }], "mlir::FunctionType", "getFunctionType">,

    /// Parameters.
    InterfaceMethod<[{
      Return the input parameter declarations live in the decl body.
    }], "ParamDeclArrayAttr", "getParamDeclsAttr">,
    InterfaceMethod<[{
      Return the result parameter declarations returned from the decl body.
    }], "TypeArrayAttr", "getResultParamTypesAttr">,
    InterfaceMethod<[{
      Return the list of input parameters.
    }], "ArrayRef<ParamDeclAttr>", "getInputParamDecls", (ins),
        "return getParamDecls($_op);", "return getParamDecls($_op);">,
    InterfaceMethod<[{
      Return the list of output parameter types.
    }], "ArrayRef<Type>", "getResultParamTypes">,

    /// Constraints.
    InterfaceMethod<[{
      Return the list of constraints.
    }], "ArrayRef<ConstraintAttr>", "getConstraints", (ins), "", [{
      return {};
    }]>,
    InterfaceMethod<[{
      Return the list of constraints.
    }], "ConstraintArrayAttr", "getConstraintsAttr", (ins), "", [{
      return ConstraintArrayAttr();
    }]>,
    InterfaceMethod<[{
      Set the constraint attribute.
    }], "void", "setConstraintsAttr",
       (ins "::M::KGEN::ConstraintArrayAttr":$attr), "", [{
      llvm_unreachable("by default ops don't have input parameters to add "
                       "constraints to");
    }]>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
