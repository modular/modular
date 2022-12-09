//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
#define KGEN_KGENDIALECT_KGENDECLINTERFACE_TD

include "mlir/IR/OpBase.td"

def DeclInterface : OpInterface<"DeclInterface"> {
  let description = [{
    This interface represents an operation whose region defines a new
    parameter scope. A parameter scope contains a set of operations where a
    particular parameter can be defined at most once and where operations may
    only reference parameters declared in that scope or in a higher.
    Declarations can shadow declarations in higher scopes.

    Parameter scopes can be isolated from above, meaning that operations nested
    within and below a scope cannot reference parameters defined in higher
    scopes, if any. An operation that defines a parameter scope must be able
    to carry constraints and can optionally define new parameters. Circular
    references are only allowed between a `DeclInterface` and itself, which
    allows, for example, functions to define parameters and then use them in
    their signatures.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the declaration scope is parametrically isolated from
      above, meaning that operations at and below the scope cannot reference
      parameters from higher scopes, if any.

      Note that `false` does not mean the scope has references to parameters in
      higher scopes but that it *can* have such references.
    }], "bool", "isIsolatedFromAbove", (ins), [{}], [{ return false; }]>,

    InterfaceMethod<[{
      Notify a declaration scope that allows references to parameters in higher
      scopes that it has no such references in its body. An implementing
      operation may store this knowledge in a state for optimizations.
    }], "void", "notifyKnownIsolatedFromAbove", (ins), [{}], [{ }]>,

    InterfaceMethod<[{
      Return the parameters declared at this parameter scope. The
      `DeclInterface` is allowed to have a circular reference to these
      parameters only.
    }], "ParamDeclArrayAttr", "getInputParamDeclsAttr">,

    InterfaceMethod<[{
      Return the constraints on the input parameters at this `DeclInterface`
      operation.
    }], "ArrayRef<ConstraintAttr>", "getConstraints", (ins), [{}], [{
      return {};
    }]>,

    InterfaceMethod<[{
      Set the constraint attribute.
    }], "void", "setConstraints",
    (ins "ArrayRef<ConstraintAttr>":$constraints), [{}], [{
      llvm_unreachable("by default ops don't have input parameters to add "
                       "constraints to");
    }]>,
  ];
}

def FuncInterface : OpInterface<"FuncInterface"> {
  let description = [{
    This interface represents an operation that defines a KGEN function. A KGEN
    function has a `SignatureType`, meaning that it can accept arguments and
    input parameters and produce results and output parameters, and has calling
    conventions. Consequently, operations that implement `FuncInterface` must
    also implement `DeclInterface`.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the function's parameter and SSA value signature.
    }], "SignatureType", "getSignature", (ins)>,
  ];

  // FIXME: Remove this when FunctionOpInterface cleanup lands.
  let extraTraitClassDeclaration = [{
    /// Return the `FunctionType` for input/result SSA values.
    FunctionType getFunctionType() { return $_op.getSignature().getValues(); }
    /// Return the `FunctionType` attribute for input/result SSA values.
    TypeAttr getFunctionTypeAttr() { return
      TypeAttr::get($_op.getSignature().getValues());
    }
  }];
  let extraSharedClassDeclaration = [{
    /// Return the function's signature including contextual parameters from
    /// higher scopes.
    SignatureType getFullSignature() { return KGEN::getFullSignature($_op); }

    /// Return the function and value argument conventions.
    DenseI8ArrayAttr getConventions() {
      return $_op.getSignature().getConventions();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type>  getArgumentTypes() {
      return $_op.getSignature().getValueInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return $_op.getSignature().getValueResults();
    }

    /// Return the input parameter declarations that live in the function body.
    ParamDeclArrayAttr getInputParamDeclsAttr() {
      return $_op.getSignature().getInputParams();
    }
    /// Return the list of input parameters.
    ArrayRef<ParamDeclAttr> getInputParamDecls() {
      return getInputParamDeclsAttr().getValue();
    }

    /// Return the result parameter declarations returned from the function
    /// body.
    TypeArrayAttr getResultParamTypesAttr() {
      return $_op.getSignature().getResultParamTypes();
    }
    /// Return the list of output parameter types.
    ArrayRef<Type> getResultParamTypes() {
      return getResultParamTypesAttr().getValue();
    }
  }];

  let dependentTraits = [DeclInterface];
}

#endif // KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
