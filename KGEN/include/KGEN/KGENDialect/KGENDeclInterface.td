//===- KGENDeclInterface.td -----------------------------------------------===//
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
    kgen.generator.interface, as well as hlkgen.generator.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    /// Parameters.
    InterfaceMethod<[{
      Return the list of input and output parameters.
    }], "std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>",
        "getParameterInfo">,
    InterfaceMethod<[{
      Return the list of input parameters.
    }], "ArrayRef<ParamDeclAttr>", "getInputParamDecls", (ins), 
        "return getParamDecls($_op);", "return getParamDecls($_op);">,
    InterfaceMethod<[{
      Return the list of output parameters.
    }], "ArrayRef<ParamDeclAttr>", "getResultParamDecls">,
    /// Constraints.
    InterfaceMethod<[{
      Return the list of constraints.
    }], "ArrayRef<ConstraintAttr>", "getConstraints">,
    InterfaceMethod<[{
      Return the list of constraints.
    }], "ConstraintArrayAttr", "getConstraintsAttr">,
    InterfaceMethod<[{
      Set the constraint attribute.
    }], "void", "setConstraintsAttr",
       (ins "::M::KGEN::ConstraintArrayAttr":$attr)>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
