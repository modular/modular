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
      Return the parameters declared at this parameter scope. These parameters
      are visible to the operations at or below the parameter scope defined by
      this operation. The `DeclInterface` is allowed to have a circular
      reference to these parameters only.
    }], "ArrayRef<ParamDeclAttr>", "getInputParamDecls">,

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

  let extraTraitClassDeclaration = [{
    /// Verify the current parameter scope if it is the top-level scope.
    LogicalResult verifyIfTopLevel(SymbolTableCollection &symtab) {
      return impl::verifyIfTopLevel(cast<DeclInterface>(*$_op), symtab);
    }
  }];
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
    InterfaceMethod<[{
      Return whether the function should be always_inline'd, and whether it
      should maintain debug info when inlined.
    }], "AlwaysInlineLevel", "getAlwaysInlineLevel", (ins)>,
  ];

  let extraSharedClassDeclaration = [{
    /// Return the function's signature including contextual parameters from
    /// higher scopes.
    SignatureType getFullSignature() { return KGEN::getFullSignature($_op); }

    /// Return the function and value argument conventions.
    ConventionsAttr getConventions() {
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
    ParamDeclArrayAttr getResultParamsAttr() {
      return $_op.getSignature().getResultParams();
    }
    /// Return the list of output parameters.
    ArrayRef<ParamDeclAttr> getResultParams() {
      return getResultParamsAttr().getValue();
    }

    /// Return the `FunctionType` for input/result SSA values.
    FunctionType getFunctionType() {
      return $_op.getSignature().getValues();
    }

    /// Return true if the function throws.
    bool isThrows() {
      return $_op.getSignature().isThrows();
    }
    /// Return true if the function is async.
    bool isAsync() {
      return $_op.getSignature().isAsync();
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Implementation
    //===------------------------------------------------------------------===//

    TypeAttr getFunctionTypeAttr() { return TypeAttr::get(getFunctionType()); }
    void setFunctionTypeAttr(TypeAttr type) {
      llvm_unreachable("can't opaquely set the function type");
    }
    ArrayAttr getArgAttrsAttr() { return {}; }
    ArrayAttr getResAttrsAttr() { return {}; }
    Attribute removeArgAttrsAttr() { llvm_unreachable("no argument attrs"); }
    Attribute removeResAttrsAttr() { llvm_unreachable("no result attrs"); }
    void setArgAttrsAttr(ArrayAttr attrs) {
      llvm_unreachable("no argument attrs");
    }
    void setResAttrsAttr(ArrayAttr attrs) {
      llvm_unreachable("no result attrs");
    }
  }];

  let dependentTraits = [DeclInterface];
}

def KGEN_CallOpInterface : OpInterface<"KGENCallOpInterface"> {
  let description = [{
    This interface represents a call to a parametric KGEN declaration: either a
    function, generator, or interface.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the parameter expression of the operation's callee.
    }], "TypedAttr", "getCallee", (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<[{
      Get the values of the parameters passed to the callee's input parameters.
    }], "ArrayRef<ParamBindAttr>", "getParamValues", (ins), [{}],
    [{ return {}; }]>,
    InterfaceMethod<[{
      Returns whether this call flavour is allowed to exist post-elaboration as
      a child of a `kgen.func` operation. If so, its callee's signature must
      have no input and output parameters.
    }], "bool", "isAllowedInFunc", (ins), [{}], [{ return true; }]>,
    InterfaceMethod<[{
      Return the declarations of the result parameters bound to the result
      parameters of the callee.
    }], "ArrayRef<ParamDeclAttr>", "getParamDecls", (ins), [{}],
    [{ return {}; }]>
  ];

  let extraSharedClassDeclaration = [{
    /// Get the signature type of the callee.
    SignatureType getCalleeType() {
      return cast<SignatureType>(getCallee().getType());
    }
  }];

  let verify = [{ return impl::verifyCallOp($_op); }];
  let verifyWithRegions = 1;
}

#endif // KGEN_KGENDIALECT_KGENDECLINTERFACE_TD
