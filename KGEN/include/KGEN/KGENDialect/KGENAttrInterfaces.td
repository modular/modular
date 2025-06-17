//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
#define KGEN_KGENDIALECT_KGENATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

def ParameterAttrInterface : AttrInterface<"ParameterAttr"> {
  let description = [{
    Any attribute that implements `TypedAttr` can be used as a parameter
    attribute in KGEN, but this interface allows parameter attributes to plug
    into specific parts of the KGEN parameter system.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Returns whether the attribute can be considered a simple constant.
    }], "bool", "isConstant">,

    InterfaceMethod<[{
      Compare this parameter attribute with another attribute of the same kind,
      returning true if this parameter should be considered "less" than the
      other attribute.
    }], "bool", "isLessThan", (ins "Attribute":$rhs), [{}], [{
      return false;
    }]>,

    InterfaceMethod<[{
      Returns a custom error message for this parameter attribute, or success if
      there are no errors. This allows parameter attributes to implement their
      own validation logic and error reporting, producing better elaborator
      error messages when they cannot be simplified then.
    }], "ErrorOrSuccess", "validateForElaborator", (ins), [{}], [{
      return success();
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return true if the parameter can be considered a "simple" constant. A
    /// parameter is a constant if it contains no nested expressions that have
    /// not been evaluated.
    static bool isSimpleConstant(Attribute attr);

    /// Compare two parameter constants, returning true if the `lhs` parameter
    /// should be considered "less" than the `rhs` parameter.
    static bool compare(Attribute lhs, Attribute rhs);
  }];
}

//===----------------------------------------------------------------------===//
// ParameterScopeAttrInterface
//===----------------------------------------------------------------------===//

def ParameterScopeAttrInterface : AttrInterface<"ParameterScopeAttrInterface"> {
  let description = [{
    The `ParameterScopeAttrInterface` describes an attribute that declares a
    nested parameter scope within a parameter expression. It enables
    `ParamIndexRefAttr` values inside the attribute to reference parameters
    declared in a scope.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the types of the nested input parameter declarations.
    }], "ArrayRef<Type>", "getInputParamTypes">,
  ];
}

//===----------------------------------------------------------------------===//
// ContextuallyEvaluatedAttrInterface
//===----------------------------------------------------------------------===//

def ContextuallyEvaluatedAttrInterface :
    AttrInterface<"ContextuallyEvaluatedAttrInterface"> {
  let description = [{
    This interface describes parameter attributes whose evaluation may require
    additional context. This is in contrast to "simple" parameter attributes
    that can be simplified context-free at construction time.
  }];
}

//===----------------------------------------------------------------------===//
// IndexRefAttrInterface
//===----------------------------------------------------------------------===//

def IndexRefAttrInterface : AttrInterface<"IndexRefAttrInterface"> {
  let description = [{
    Index-based parameter references are a relative parameter referencing scheme
    that uses a pair of integers to reference parameters in a way that doesn't
    involve names. This is useful for later knowing if two types are equal, even
    if they have different parameter names.

    For example, these two aliases have equal types:

    ```mojo
    alias A: fn[T: AnyType](x: T)->None = ...
    alias B: fn[Y: AnyType](x: Y)->None = ...
    ```

    ...if those param-refs use indexes instead of names like:

    ```mojo
    alias A: fn[_: AnyType](x: *(0,0))->None = ...
    alias B: fn[_: AnyType](x: *(0,0))->None = ...
    ```

    All types in Mojo use `IndexRefAttrInterface` instead of parameter names.
    The above are `ParamIndexRefAttr` specifically.

    All `IndexRefAttrInterface` have two fields: a depth and an index.

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature
      (like above), one means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative integer).

    See IRAIDAI for more details, context, and examples.

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Get the depth of the index reference.
    }], "size_t", "getDepth">,
    InterfaceMethod<[{
      Get the index of the referenced parameter.
    }], "size_t", "getIndex">,
    InterfaceMethod<[{
      Change the depth and index of the parameter without changing the semantics
      of the parameter it refers to. This method also supplies the new
      subelements of the attribute.
    }], "IndexRefAttrInterface", "replace",
      (ins "size_t":$depth, "size_t":$index, "ArrayRef<Attribute>":$attrs,
           "ArrayRef<Type>":$types)>
  ];
}

//===----------------------------------------------------------------------===//
// DeclRefAttrInterface
//===----------------------------------------------------------------------===//

def DeclRefAttrInterface : AttrInterface<"DeclRefAttrInterface"> {
  let description = [{
    This interface provides a mechanism for attributes of `!lit.struct` type
    to implement a verifier hook. `StructType` is a reference to an out-of-line
    type definition and values of this type require a symbol table to lookup the
    definition for verification.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify whether the value of the attribute matches the type. A symbol table
      evaluation handler is provided to lookup the type definition and perform
      any parameter evaluation necessary.
    }], "LogicalResult", "verifySymbolUses",
      (ins "SymTabEvaluationContext &":$evaluationContext,
           "Location":$loc)>
  ];
}

//===----------------------------------------------------------------------===//
// FnMetadataAttrInterface
//===----------------------------------------------------------------------===//

def FnMetadataAttrInterface : AttrInterface<"FnMetadataAttrInterface"> {
  let description = [{
    This interface describes attributes that are attached to a `!kgen.func`
    type. Function metadata attributes carry additional information about a
    callable on top of the information in the base `FuncType`. This
    interface defines the required methods for this metadata attribute,
    including verification and print hooks.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify the FuncType parameters using the current metadata.
    }], "LogicalResult", "verifyFuncType", (ins
      "function_ref<InFlightDiagnostic()>":$emitError,
      "FunctionType":$values,
      "ArrayRef<ArgConvention>":$argConventions,
      "FnEffects":$effects)>,
    InterfaceMethod<[{
      Request the custom metadata attribute print the provided FuncType.
    }], "void", "printFuncType", (ins "AsmPrinter &":$p,
                                          "FuncType":$sig)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a FuncType where the first `numBound`
      positional arguments have been removed from the FuncType by being
      explicitly bound to values, such as in closure formation.
    }], "FnMetadataAttrInterface", "getWithBoundPosArgs", (ins
      "size_t":$numBound)>,
  ];
}

//===----------------------------------------------------------------------===//
// GeneratorMetadataAttrInterface
//===----------------------------------------------------------------------===//

def GeneratorMetadataAttrInterface
    : AttrInterface<"GeneratorMetadataAttrInterface"> {
  let description = [{
    This interface describes attributes that are attached to a GeneratorType,
    and carries additional metadata about the list. This interface defines the
    required methods for this metadata attribute, including verification and
    print hooks.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify the generator using the current metadata.
    }], "LogicalResult", "verifyGenerator", (ins
      "function_ref<InFlightDiagnostic()>":$emitError,
      "ArrayRef<Type>":$inputParamTypes, "Type":$body)>,
    InterfaceMethod<[{
      Request the custom metadata attribute print the provided generator.
    }], "void", "printGenerator", (ins "AsmPrinter &":$p,
                                       "GeneratorType":$generator)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a generator where the parameters
      indicated by set bits in `boundParams` been removed from the generator by
      being explicitly bound to values, such as those in call signatures.
    }], "GeneratorMetadataAttrInterface", "getWithBoundParams", (ins
      "const llvm::BitVector &":$boundParams)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a generator with a number of positional
      input parameters prepended, determined from the number of declared by the
      given ops. The given operations must implement `DeclInterface`.
    }], "GeneratorMetadataAttrInterface", "prependPosParamsFromOps", (ins
      "ArrayRef<Operation *>":$ops)>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
