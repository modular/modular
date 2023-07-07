//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_TD
#define KGEN_KGENDIALECT_KGENATTRS_TD

include "KGEN/KGENDialect/KGENAttrInterfaces.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class AttrOfType<Type type> : ConfinedAttr<TypedAttrInterface, [
  AttrConstraint<SubstLeaves<"$_self", "$_self.cast<TypedAttr>().getType()",
                             type.predicate>, type.summary>]>;

/// An attribute in the KGEN dialect.
class KGENAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KGENDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// The base class of all parameter expression attributes. Expression attributes
/// are all typed attributes.
class KGEN_ExprAttr<string attrName, string attrMnemonic,
                    list<Trait> traits = []>
    : KGENAttr<attrName, attrMnemonic,
               [TypedAttrInterface,
                DeclareAttrInterfaceMethods<ParameterAttrInterface>] # traits>;

//===----------------------------------------------------------------------===//
// ArrayOfAttrsAttr
//===----------------------------------------------------------------------===//

/// An array attribute of other attributes. This one supports sub-element
/// interface.
class KGEN_ArrayOfAttrsAttr<string attrName, string attrMnemonic,
                            string eltName>
    : ArrayOfAttr<KGENDialect, attrName, attrMnemonic, eltName>;

//===----------------------------------------------------------------------===//
// ParameterExprArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_ParameterExprArrayAttr
    : KGEN_ArrayOfAttrsAttr<"ParameterExprArray", "exprs", "TypedAttr">;

//===----------------------------------------------------------------------===//
// StringArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_StringArrayAttr
    : ArrayOfAttr<KGENDialect, "StringArray", "strings", "StringAttr">;

//===----------------------------------------------------------------------===//
// TypeArrayAttr
//===----------------------------------------------------------------------===//

// This is an array of type objects, stored in a dense representation that can
/// be turned into an ArrayRef.
/// NOTE: We have to define this as TypeArrayAttr because ODS squats on the
/// `TypeArrayAttr` name with something different.
def KGEN_TypeArrayAttr
    : KGEN_ArrayOfAttrsAttr<"TypeArray", "type.array", "Type">;

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

/// Declaration of a named parameter.
def KGEN_ParamDeclAttr : KGENAttr<"ParamDecl", "param.decl"> {
  let summary = "Parameter declaration";
  let description = [{
    This is a declaration of a parameter in the meta-programming domain for
    generators and related infrastructure.  These are typically owned by
    kgen.generator instances or other things that produce new attributes.
  }];

  /// ParamDeclAttr's always have a `name` and `type`: an omitted type in the
  /// asmsyntax defaults to `index`.
  let parameters = (ins "StringAttr":$name, "Type":$type);

  let assemblyFormat = "custom<ParamName>($name) `:` $type";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type),
      "return $_get(type.getContext(), name, type);">,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">
  ];
}

//===----------------------------------------------------------------------===//
// ParamDeclArrayAttr
//===----------------------------------------------------------------------===//

/// An array of ParamDeclAttr's typically used on a kgen.generator and other
/// ops that define parameters.
def KGEN_ParamDeclArrayAttr
    : KGEN_ArrayOfAttrsAttr<"ParamDeclArray", "param.decls", "ParamDeclAttr"> {
  let builders = [
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$paramDecl), [{
      return get(paramDecl.getContext(), paramDecl);
    }]>
  ];
}

/// An array of ParamDeclAttr of size 1.
def ParamDeclSize1ArrayAttr
  : ConfinedAttr<KGEN_ParamDeclArrayAttr, [AttrConstraint<
      CPred<"$_self.cast<ParamDeclArrayAttr>().getValue().size() == 1">,
      "single parameter declaration"
    >]>;

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamBindAttr : KGENAttr<"ParamBind", "param.bind"> {
  let summary = "Bind a value to a parameter declaration.";
  let description = [{
    The `#kgen.param.bind` attribute binds a concrete value to a named
    parameter, for example a `kgen.return` uses it to bind returned parameters
    to the enclosing generator's parameter declaration. Call operations use
    it to bind values to the input parameters of the callee.

    Example:

    ```mlir
    #kgen<param.bind foo: dtype = f32>
    ```
  }];

  let parameters = (ins "StringAttr":$name, "TypedAttr":$value);

  let assemblyFormat = "custom<ParamName>($name) `=` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "TypedAttr":$value), [{
      return get(name.getContext(), name, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "TypedAttr":$value),
    [{
      return get(StringAttr::get(value.getContext(), name), value);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Return the type of the value being bound.
    Type getType() const { return getValue().getType(); }
  }];
}

/// An array of ParamBindAttr's.
def ParamBindArrayAttr
    : KGEN_ArrayOfAttrsAttr<"ParamBindArray", "param.binds", "ParamBindAttr">;

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

def KGEN_ConstraintAttr : KGENAttr<"Constraint", "constraint"> {
  let summary = "kgen constraint";
  let description = [{
    A constraint attribute represents a constraint on a generator or
    generator interface. It consists of a boolean parameter expression that must
    evaluate to true for the constraint to be satisfied, a message/description,
    and an MLIR location for emitting diagnostics.

    When unspecified in a `.mlir` file, the location defaults to the textual
    location of the attribute.
  }];

  let parameters = (ins "TypedAttr":$expr, "StringAttr":$message,
                        "Location":$loc);

  let assemblyFormat = [{
    `[` custom<ParamValue>($expr, "$_builder.getI1Type()") `,` $message
    `` custom<ConstraintLoc>($loc) `]`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "TypedAttr":$expr, "StringAttr":$message, "Location":$loc), [{
      return $_get(expr.getContext(), expr, message, loc);
    }]>
  ];

  let genVerifyDecl = 1;
}

/// An array of ConstraintAttr's.
def ConstraintArrayAttr
    : KGEN_ArrayOfAttrsAttr<"ConstraintArray", "constraints", "ConstraintAttr">;

//===----------------------------------------------------------------------===//
// MetadataAttr
//===----------------------------------------------------------------------===//

def KGEN_MetadataAttr : KGENAttr<"Metadata", "metadata"> {
  let summary = "Function metadata.";
  let description = [{
    The `#kgen.metadata` attribute contains additional information about a
    function, such as its argument calling conventions (pass-by-value or
    pass-by-reference), any default argument values, and the effects of the
    function itself: none or throws.

    Example:

    ```mlir
    // Metadata specifying input argument #0 is taken by reference,
    // argument #1 is taken by value and has a default value of `4.2`,
    // and the function has a "throws" function effect.
    #kgen.metadata<[byref, owned], [<1 : i64 = 4.2 : f32>], throws>

    // Metadata specifying the default input value conventions, no default
    // arguments, and a "throws" function effect.
    #kgen.metadata<[], [], throws>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"ValueInputConvention">:$inputConventions,
    OptionalArrayRefParameter<"TypedAttr">:$defaultArguments,
    "FnEffects":$fnEffects
  );
  let assemblyFormat = [{
    `<` `[` (`]`) : ($inputConventions^ `]`)? `,`
        `[` (`]`) : ($defaultArguments^ `]`)? `,`
    $fnEffects `>`
  }];

  /// These builders returns the default metadata: all input arguments are
  /// pass-by-value, and there are no function effects.
  let builders = [
    AttrBuilder<(ins "unsigned":$numInputs)>,
    AttrBuilder<(ins "unsigned":$numInputs, "FnEffects":$effects)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if all input argument conventions are pass-by-value, and
    /// there are no function effects.
    bool isDefault();

    /// Verify the metadata with respect to the enclosing signature type.
    LogicalResult verifySignature(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<Type> inputParamTypes,
                                  ArrayRef<Type> resultParamTypes,
                                  FunctionType values);

    /// Get this metadata attr with the effects swapped out.
    MetadataAttr getWithFnEffects(FnEffects effects);
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// VariadicAttr
//===----------------------------------------------------------------------===//

def KGEN_VariadicAttr : KGEN_ExprAttr<"Variadic", "variadic", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A parameterized variadic attribute.";
  let description = [{
    The `#kgen.variadic` attribute contains a homogeneous list of elements of an
    variadic type. It is used to represent constant variadic sequence values.

    Example:

    ```mlir
    #kgen.variadic<1, 2> : !kgen.variadic<index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "VariadicType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "VariadicType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

def KGEN_UnknownAttr : KGEN_ExprAttr<"Unknown", "unknown"> {
  let summary = "An unknown parameter value.";
  let description = [{
    The `#kgen.unknown` attribute represents an unknown parameter value. It is
    a special placeholder that represents a parameter that may only be known
    dynamically.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

def KGEN_UnboundAttr : KGEN_ExprAttr<"Unbound", "unbound"> {
  let summary = "An unbound parameter value.";
  let description = [{
    The `#kgen.unbound` attribute represents an unbound parameter value. It is
    a special placeholder that appears in collections of parameters that have
    been partially bound.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// This attribute models a reference to a named parameter.  The type of the
/// ParamDeclRefAttr must always be the same as the type of the parameter being
/// referenced.
def KGEN_ParamDeclRefAttr : KGEN_ExprAttr<"ParamDeclRef", "param.decl.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a parameter value.";
  let description = [{
    The `#kgen.param.decl.ref` attribute is a typed attribute that represents
    a reference to a declared parameter. It contains the type of the referenced
    parameter and its name.

    Example:

    ```mlir
    // A reference to parameter "p" with type "i1".
    #kgen.param.decl.ref<"p"> : i1
    ```
  }];

  let parameters = (ins "StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type), [{
      return get(name.getContext(), name, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type), [{
      return get(StringAttr::get(type.getContext(), name), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$decl), [{
      return get(decl.getName(), decl.getType());
    }]>
  ];

  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamIndexRefAttr : KGEN_ExprAttr<"ParamIndexRef", "param.index.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a signature input or result parameter.";
  let description = [{
    The `#kgen.param.index.ref` attribute is a reference to an input or result
    parameter of an enclosing signature. This attribute can only be used inside
    a `SignatureType`. This attribute contains three fields: the depth of the
    referenced signature relative to the nearest signature (non-negative
    integer), a flag indicating input or result parameter, and the index of the
    parameter (non-negative integer).

    Example:

    ```mlir
    // Second input parameter of the nearest signature.
    #kgen.param.index.ref<0, false, 1> : index
    // First result parameter of next enclosing signature.
    #kgen.param.index.ref<1, true, 0> : !kgen.declref<@Int>
    ```
  }];

  let parameters = (ins "size_t":$depth, "bool":$isResult, "size_t":$index,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $depth `,` $isResult `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "size_t":$depth, "bool":$isResult,
                                        "size_t":$index, "Type":$type), [{
      return $_get(type.getContext(), depth, isResult, index, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ConcreteTypeConstantAttr
//===----------------------------------------------------------------------===//

def KGEN_ConcreteTypeConstantAttr : KGEN_ExprAttr<"ConcreteTypeConstant",
    "concretetype.constant"> {
  let summary = "Constant !kgen.type MLIR type without parameter references";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute that refers to a type.  The difference
    is that it is a `TypedAttr` so it can be a parameter expression, and has
    'kgen.type' itself.

    The bound type must not have type parameters.
  }];

  // Inherit from a base class.
  let cppBaseClassName = "TypeConstantAttr";

  let parameters = (ins
    "Type":$value,
    AttributeSelfTypeParameter<"", "mlir::Type",
                               "MLIRTypeType::get($_ctxt)">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$value), "", "TypedAttr">,
  ];

  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ParameterizedTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This represents a parameter whose value is an MLIR type.  It is like
/// `TypeAttr` (an attribute referring to a type) but is a `TypedAttr` that
/// refers to an MLIR type and has 'kgen.type' itself.
def KGEN_ParameterizedTypeConstantAttr : KGEN_ExprAttr<
    "ParameterizedTypeConstant", "parameterizedtype.constant"> {
  let summary = "Constant !kgen.type MLIR type with parameter references";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute refers to a type.  The difference is
    that it is a `TypedAttr` so it can be a parameter expression, and has
    'kgen.type' itself.

    The bound type is known to have type parameters.
  }];

  // Inherit from a base class and re-use ConcreteTypeConstantAttr's storage.
  let cppBaseClassName = "TypeConstantAttr";
  let storageClass = KGEN_ConcreteTypeConstantAttr.storageClass;
  let genStorageClass = 0;

  let parameters = (ins
    "Type":$value,
    AttributeSelfTypeParameter<"", "mlir::Type",
                               "MLIRTypeType::get($_ctxt)">:$type
  );

  let builders = [
    AttrBuilder<(ins "mlir::Type":$value), "", "TypedAttr">,
    AttrBuilder<(ins "mlir::Type":$value, "mlir::Type":$resultType), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$value), "", "TypedAttr">,
  ];

  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def KGEN_DTypeConstantAttr : KGEN_ExprAttr<"DTypeConstant", "dtype.constant",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to DType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "KGENDType":$dType,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [AttrBuilder<(ins "KGENDType":$dtype)>];

  let assemblyFormat = "`<` $dType `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Check if the DType is convertible to the MLIR type.
    bool isConvertibleTo(Type type);
    /// Check if the MLIR type can be converted from the DType.
    bool isConvertibleFrom(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

def KGEN_SymbolConstantAttr : KGEN_ExprAttr<"SymbolConstant", "symbol.constant",
    [DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "kgen.func or kgen.generator reference";
  let description = [{
    This is a value of !kgen.signature type, which refers to a func or
    generator.  This may optionally bind the input parameter values at time of
    formation - when this happens, the result type is non-parametric.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        OptionalArrayRefParameter<"TypedAttr">:$paramValues,
                        AttributeSelfTypeParameter<"", "SignatureType">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "SignatureType":$type), [{
      return get(type.getContext(), symbol, {}, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "SignatureType":$type), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,
  ];

  let assemblyFormat = "`<` $symbol custom<ParameterValues>($paramValues) `>`";
}

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

def KGEN_TargetParamAttr : KGEN_ExprAttr<"TargetParam", "target"> {
  let summary = "Target configuration parameter attribute.";
  let description = [{
    The `#kgen.target` is an attribute representing a target of type
    `!kgen.target`. It contains the target configuration information.

    Example:

    ```mlir
    #kgen.target<triple="triple", cpu="cpu", features="features",
                 data_layout="p:32:32", simd_bit_width=128>
    ```

    Target types can be manipulated using target operators.

    Example:

    ```mlir
    kgen.generator @target_host<t0: target>()
        constraints <[eq(:target
            #kgen.target<triple="triple", cpu="cpu", features="features",
                         data_layout="", simd_bit_width=128>,
            t0),
          "Must support the target!"
        ]> {
      kgen.return
    }
    ```
  }];

  let parameters = (ins "TargetInfoAttr":$target,
                         AttributeSelfTypeParameter<"", "TargetType">:$type);

  let builders =[
    AttrBuilderWithInferredContext<(ins "TargetInfoAttr":$target,
                                        "TargetType":$type), [{
      return get(type.getContext(), target, type);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// BuildInfoParamAttr
//===----------------------------------------------------------------------===//

def KGEN_BuildInfoParamAttr : KGEN_ExprAttr<"BuildInfoParam", "build_info"> {
  let summary = "Build configuration parameter attribute.";
  let description = [{
    The `#kgen.build_info` is an contains the build configuration information.

    There are two textual formats that represent this attribute:

    ```mlir
    #kgen<build_info host>
    #kgen.build_info<type="debug">
    ```

    `#kgen<build_info>` is sugar for the build parameter corresponding to the
    build configuration of the modular codebase. It will be fully resolved when
    printed, and therefore it will NOT be printed as `kgen<build_info>`.

    Example:

    ```mlir
    kgen.generator @build_is_debug()
      constraints <[
        eq(:string build_info_get_field(#kgen<build_info host>, "type"),
                   "debug"),
        "Must be built in release mode!"]> {
      kgen.return
    }

    kgen.generator @build_is_debug()
      constraints <[
        eq(:string build_info_get_field(#kgen.build_info<type="debug">, "type"),
                   "release"),
        "Must be built in release mode!"]> {
      kgen.return
    }
    ```
  }];

  let parameters = (ins "BuildInfoAttr":$buildInfo,
                        AttributeSelfTypeParameter<"", "BuildInfoType">:$type);

  let builders =[
    AttrBuilderWithInferredContext<(ins "BuildInfoAttr":$buildInfo,
                                        "BuildInfoType":$type), [{
      return get(type.getContext(), buildInfo, type);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// EnvAttr
//===----------------------------------------------------------------------===//

def KGEN_EnvAttr : KGENAttr<"Env", "env"> {
  let summary = "Environment variable attribute.";
  let description = [{
    The `#kgen.env` attribute defines a generic dictionary of environment
    parameters that can be accessed through parameter operators. The values
    contained can be integers, represented with the `index` type, unit, the
    presence of which indicates something, and strings, represented as
    `!kgen.string` types.

    Example:

    ```mlir
    #kgen<env{intVal = 1 : index, unitAttr, strVal = "hello" : !kgen.string}>
    ```
  }];

  let parameters = (ins "DictionaryAttr":$values);
  let assemblyFormat = "`<` $values `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "DictionaryAttr":$values), [{
      return $_get(values.getContext(), values);
    }]>
  ];

  let extraClassDeclaration = [{
    /// This is the attribute name of the `#kgen.env` attribute attached to the
    /// surrounding `ModuleOp`.
    static llvm::StringLiteral getEnvAttrName() { return "kgen.env"; }

    /// Parse a defines list and create the environment from that.
    static ErrorOr<EnvAttr> parseDefines(MLIRContext *ctx,
                                         ArrayRef<std::string> defines);
  }];
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

// ParamOperatorAttr forms the core of the expression grammar for parameter
// expressions.  Its operators have been designed so that expression building
// force a deterministic canonical form on expression trees.  This ensures that
// `(x+1)*4` and `(x*4)+4` canonicalize into the same pointer value.
//
// For this reason, we should never have unary negate or subtract. `(x-y)` is
// modeled as `(x+-y)` => `(x + -1*y)`.  The ParamOperatorAttr class does
// provide convenience methods for these aggregate operations.
//
// The design of these operators assumes signful types like si64, and the fully
// associative operations are take arbitrary arity, also to make
// canonicalization simpler.
def KGEN_ParamOperatorAttr : KGEN_ExprAttr<"ParamOperator", "param.expr", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Parameter expression";
  let parameters = (ins "POC":$opcode,
                        OptionalArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"">:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins
      "POC":$opcode, "ArrayRef<TypedAttr>":$operands,
      "mlir::Type":$type
    ), "", "TypedAttr">,

    // Build a parameter expression.  This automatically canonicalizes and
    // folds, so it may not necessarily return a ParamOperatorAttr.
    AttrBuilderWithInferredContext<(ins
      "POC":$opcode, "ArrayRef<TypedAttr>":$operands
    ), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins
        "POC":$opcode, "ArrayRef<TypedAttr>":$operands, "Type":$type), [{
      return get(type.getContext(), opcode, operands, type);
    }], "TypedAttr">,

    // Build a binary parameter expression for convenience.
    AttrBuilderWithInferredContext<(ins "POC":$opcode, "TypedAttr":$lhs,
                                        "TypedAttr":$rhs), [{
      TypedAttr operands[] = { lhs, rhs };
      return get(opcode, operands);
    }], "TypedAttr">
  ];

  let assemblyFormat = "`<` $opcode (`,` $operands^)? `>`";
  let genVerifyDecl = 1;  // Generate getChecked/verify.

  let extraClassDeclaration = [{
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }

    /// Return (ne x, y) which is the same as (not (eq x, y)).
    static TypedAttr getNE(TypedAttr lhs, TypedAttr rhs) {
      return getNot(get(POC::EQ, lhs, rhs));
    }

    /// Return (not x) which is the same as (xor x, true).  The `operand` value
    /// must have type `i1`.
    static TypedAttr getNot(TypedAttr operand);

    /// Return (neg x) which is the same as (mul x, -1).  The `operand` value
    /// must have `index` type.
    static TypedAttr getNeg(TypedAttr operand);

    /// Return (x-y) which is the same as (add x, (neg y)).  The `operand` value
    /// must have `index` type.
    static TypedAttr getSub(TypedAttr lhs, TypedAttr rhs);
  }];
}

//===----------------------------------------------------------------------===//
// MLIROpAttr
//===----------------------------------------------------------------------===//

def MLIROpAttr : KGEN_ExprAttr<"MLIROp", "param.mlir_op"> {
  let summary = "an MLIR operation as a parameter expression";
  let description = [{
    The `#kgen.param.mlir_op` attribute represents an MLIR operation as a
    parameter expression. Its type is signature type.

    Example:

    ```
    #kgen.param.mlir_op<"index.add", {}>
      : !kgen.signature<(index, index) -> index>
    ```

    Operation attributes can be specified using a dictionary attribute.

    Example:

    ```
    #kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>
      : !kgen.signature<(index, index) -> i1>
    ```

    The operation can be parameterized on any of its attributes that are
    parametric -- that is, which are `TypedAttr`. These attributes are omitted
    from the attribute dictionary and are present in the signature.

    Example:

    ```
    #kgen.param.mlir_op<"pop.array.get", {}> : !kgen.signature<
      <*"index">(!pop.array<size, type>) -> !kgen.paramref<type>
    >
    ```

    Parametric operations can be bound using `bind_signature`.
  }];

  let parameters = (ins "StringAttr":$name, "DictionaryAttr":$attrs,
                        AttributeSelfTypeParameter<"", "SignatureType">:$type);

  let assemblyFormat = "`<` $name `,` $attrs `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name, "DictionaryAttr":$attrs, "SignatureType":$type), [{
      return $_get(name.getContext(), name, attrs, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// KGEN_RegionAttr
//===----------------------------------------------------------------------===//

def KGEN_RegionAttr
    : KGEN_ExprAttr<"Region", "param.region"> {
  let summary = "region to be used as a local parameter";
  let description = [{
    This attribute stores the necessary attributes to allow a region to
    participate in normal parameter resolution. A region referenced
    by an attribute of this kind may have additional restrictions
    based on whether or not it captures SSA values - the caller is
    responsible for ensuring those restrictions are fulfilled.
  }];
  let parameters = (ins "StringAttr":$regionName,
                        OptionalArrayRefParameter<"TypedAttr">:$paramValues,
                        "BoolAttr":$isolated,
                        AttributeSelfTypeParameter<"", "SignatureType">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$decl, "bool":$isolated)>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "BoolAttr":$isolated,
                                        "SignatureType":$type)>,
  ];

  let assemblyFormat = [{
    `<` $regionName `,` `[` custom<ParameterValues>($paramValues) `]` `,`
    `isolated` `=` $isolated `>`
  }];
}

#endif // KGEN_KGENDIALECT_KGENATTRS_TD
