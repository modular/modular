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
      Compare this parameter attribute with another attribute that may not be of
      the same kind, returning true if this parameter should be considered
      "less" than the other attribute. Returns `std::nullopt` if no viable
      ordering could be established.

      Parameters are sorted between those that are simple constants and those
      that aren't, meaning that if `lhs` is a simple constant, then `rhs` will
      also be a simple constant, and if `lhs` is not a simple constant, then
      `rhs` won't be either.
    }], "std::optional<bool>", "isLessThan", (ins "Attribute":$rhs), [{}], [{
      return std::nullopt;
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
    nested parameter scope within an attribute expression. It enables
    `ParamIndexRefAttr` values inside the type to reference parameters declared
    in a scope.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Return the types of the nested input parameter declarations.
    }], "ArrayRef<Type>", "getInputParamTypes">,
    InterfaceMethod<[{
      Return the types of the nested result parameter declarations.
    }], "ArrayRef<Type>", "getResultParamTypes">
  ];
}


//===----------------------------------------------------------------------===//
// IndexRefAttrInterface
//===----------------------------------------------------------------------===//

def IndexRefAttrInterface : AttrInterface<"IndexRefAttrInterface"> {
  let description = [{
    Index-based parameter references are a relative parameter referencing scheme
    that uses a pair of integers to reference parameters in a name-agnostic way.
    They contain a depth and an index. The depth indicates how many parameter
    scopes exist between the parameter being referenced and the reference
    itself. The index is simply the parameter number at the referenced scope.

    This interace allows other kinds (see `LIT::ImplicitLifetimeRefAttr`) of
    index-based references to interplay with parts of the KGEN parameter system.
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
      is provided to lookup the type definition.
    }], "LogicalResult", "verifySymbolUses",
      (ins "Operation *":$module, "mlir::LockedSymbolTableCollection &":$symtab,
           "Location":$loc)>
  ];
}

//===----------------------------------------------------------------------===//
// FnMetadataAttrInterface
//===----------------------------------------------------------------------===//

def FnMetadataAttrInterface : AttrInterface<"FnMetadataAttrInterface"> {
  let description = [{
    This interface describes attributes that are attached to a `!kgen.signature`
    type. Function metadata attributes carry additional information about a
    callable on top of the information in the base `SignatureType`. This
    interface defines the required methods for this metadata attribute,
    including verification and print hooks.
  }];
  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<[{
      Verify the signature type parameters using the current metadata.
    }], "LogicalResult", "verifySignature", (ins
      "function_ref<InFlightDiagnostic()>":$emitError,
      "ArrayRef<Type>":$inputParamTypes, "ArrayRef<Type>":$resultParamTypes,
      "FunctionType":$values,
      "ArrayRef<ArgConvention>":$argConventions,
      "FnEffects":$effects)>,
    InterfaceMethod<[{
      Request the custom metadata attribute print the provided signature.
    }], "void", "printSignature", (ins "AsmPrinter &":$p,
                                       "SignatureType":$sig)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a signature where the parameters
      indicated by set bits in `boundParams` been removed from the signature by
      being explicitly bound to values, such as in call signatures.
    }], "FnMetadataAttrInterface", "getWithBoundParams", (ins
      "const llvm::BitVector &":$boundParams)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a signature where the first `numBound`
      positional arguments have been removed from the signature by being
      explicitly bound to values, such as in closure formation.
    }], "FnMetadataAttrInterface", "getWithBoundPosArgs", (ins
      "size_t":$numBound)>,
    InterfaceMethod<[{
      Get a new metadata attribute for a signature with a number of positional
      input parameters prepended, determined from the number of declared by the
      given ops. The given operations must implement `DeclInterface`.
    }], "FnMetadataAttrInterface", "prependPosParamsFromOps", (ins
      "ArrayRef<Operation *>":$ops)>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENATTRINTERFACES_TD
