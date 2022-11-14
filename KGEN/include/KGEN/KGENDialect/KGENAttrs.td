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

include "KGEN/KGENDialect/KGENDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

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
class ExprAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : KGENAttr<attrName, attrMnemonic, [TypedAttrInterface] # traits>;

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

/// Declaration of a named parameter.
def ParamDeclAttr : KGENAttr<"ParamDecl", "param.decl", [
    SubElementAttrInterface
  ]> {
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
// ArrayOfAttrsAttr
//===----------------------------------------------------------------------===//

/// An array attribute of other attributes. This one supports sub-element
/// interface.
class ArrayOfAttrsAttr<string attrName, string attrMnemonic, string eltName>
    : ArrayOfAttr<KGENDialect, attrName, attrMnemonic, eltName,
                  [SubElementAttrInterface]>;

//===----------------------------------------------------------------------===//
// ParameterExprArrayAttr
//===----------------------------------------------------------------------===//

def ParameterExprArrayAttr : ArrayOfAttrsAttr<"ParameterExprArray", "exprs",
                                              "TypedAttr">;

//===----------------------------------------------------------------------===//
// KStringArrayAttr
//===----------------------------------------------------------------------===//

def KStringArrayAttr : ArrayOfAttr<KGENDialect, "StringArray", "strings",
                                   "StringAttr">;

//===----------------------------------------------------------------------===//
// ParamDeclArrayAttr
//===----------------------------------------------------------------------===//

/// An array of ParamDeclAttr's typically used on a kgen.generator and other
/// ops that define parameters.
def ParamDeclArrayAttr
    : ArrayOfAttrsAttr<"ParamDeclArray", "param.decls", "ParamDeclAttr">;

/// An array of ParamDeclAttr of size 1.
def ParamDeclSize1ArrayAttr
  : ConfinedAttr<ParamDeclArrayAttr, [AttrConstraint<
      CPred<"$_self.cast<ParamDeclArrayAttr>().getValue().size() == 1">,
      "single parameter declaration"
    >]>;

//===----------------------------------------------------------------------===//
// KTypeArrayAttr
//===----------------------------------------------------------------------===//

// This is an array of type objects, stored in a dense representation that can
/// be turned into an ArrayRef.
/// NOTE: We have to define this as KTypeArrayAttr because ODS squats on the
/// `TypeArrayAttr` name with something different.
def KTypeArrayAttr
    : ArrayOfAttrsAttr<"TypeArray", "type.array", "Type">;

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

/// This attribute binds a concrete value to a named parameter, for example a
/// kgen.return uses it to bind returned parameters the the enclosing generators
/// parameter declarations.
def ParamBindAttr : KGENAttr<"ParamBind", "param.bind", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "Bind a value to a parameter declaration.";
  let parameters = (ins "ParamDeclAttr":$decl, "TypedAttr":$value);

  let assemblyFormat = "custom<ParamDecl>($decl) `=` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$decl,
                                        "TypedAttr":$value), [{
      return get(decl.getContext(), decl, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "TypedAttr":$value),
    [{
      return get(ParamDeclAttr::get(name, value.getType()), value);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Return the name of the declaration being initialized.
    StringAttr getName() const;
    // Return the type of the value being bound.
    Type getType() const { return getValue().getType(); }
  }];
  let extraClassDefinition = [{
    StringAttr ParamBindAttr::getName() const { return getDecl().getName(); }
  }];
  let genVerifyDecl = 1;
}

/// An array of ParamBindAttr's.
def ParamBindArrayAttr
    : ArrayOfAttrsAttr<"ParamBindArray", "param.binds", "ParamBindAttr">;

//===----------------------------------------------------------------------===//
// ListAttr
//===----------------------------------------------------------------------===//

def ListAttr : ExprAttr<"List", "list", [SubElementAttrInterface]> {
  let summary = "homogenous list attribute";
  let description = [{
    The `#kgen.list` attribute contains a variable number of attributes of the
    same type. The type of this attribute must always be a `!kgen.list`.

    Example:

    ```mlir
    #kgen.list<fooParam, 1, add(lhsParam, 2)> : !kgen.list<index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "ListType">:$type);
  let assemblyFormat = [{
    `<` (`>`) : (custom<ListValue>($values, ref($type))^ `>`)?
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ParamRegionRefAttr
//===----------------------------------------------------------------------===//

def ParamCallRegionRefAttr
   : ExprAttr<"ParamCallRegionRef", "param.call.region.ref"> {
  let summary =
    "A marker indicating a signature value is specified by a call region";
  let parameters = (ins AttributeSelfTypeParameter<"", "SignatureType">:$type);

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "SignatureType":$type), [{
      return get(type.getContext(), type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// This attribute models a reference to a named parameter.  The type of the
/// ParamDeclRefAttr must always be the same as the type of the parameter being
/// referenced.
def ParamDeclRefAttr : ExprAttr<"ParamDeclRef", "param.decl.ref"> {
  let summary = "A reference to a parameter value.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Type":$type), [{
      return get(name.getContext(), name, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                         "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,
  ];

  let assemblyFormat = "`<` $name `>`";
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
def ParamOperatorAttr : ExprAttr<"ParamOperator", "param.expr", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface, [
      "walkImmediateSubElements", "replaceImmediateSubElements"
    ]
  >]> {
  let summary = "Parameter expression";
  let parameters = (ins "POC":$opcode,
                        ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"">:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins
      "POC":$opcode, "llvm::ArrayRef<TypedAttr>":$operands,
      "mlir::Type":$type
    ), "", "TypedAttr">,

    // Build a parameter expression.  This automatically canonicalizes and
    // folds, so it may not necessarily return a ParamOperatorAttr.
    AttrBuilderWithInferredContext<(ins
      "POC":$opcode, "llvm::ArrayRef<TypedAttr>":$operands
    ), "", "TypedAttr">,

    // Build a binary parameter expression for convenience.
    AttrBuilderWithInferredContext<(ins "POC":$opcode, "TypedAttr":$lhs,
                                        "TypedAttr":$rhs), [{
      TypedAttr operands[] = { lhs, rhs };
      return get(opcode, operands);
    }], "TypedAttr">
  ];

  let assemblyFormat = "`<` $opcode `,` $operands `>`";
  let genVerifyDecl = 1;  // Generate getChecked/verify.

  let extraClassDeclaration = [{
    size_t getNumOperands() {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) {
      return getOperands()[i];
    }

    /// Return (ne x, y) which is the same as (not (eq x, y)).
    static TypedAttr getNE(TypedAttr lhs, TypedAttr rhs) {
      return getNot(get(POC::EQ, lhs, rhs));
    }

    /// Return (not x) which is the same as (xor x, true).  The `operand` value
    /// must have type `i1`.
    static TypedAttr getNot(TypedAttr operand);
  }];
}

//===----------------------------------------------------------------------===//
// ConcreteTypeConstantAttr
//===----------------------------------------------------------------------===//

def ConcreteTypeConstantAttr
  : ExprAttr<"ConcreteTypeConstant", "concretetype.constant", [
    SubElementAttrInterface
  ]> {
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
    AttrBuilderWithInferredContext<(ins "mlir::Type":$value)>,
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
def ParameterizedTypeConstantAttr
  : ExprAttr<"ParameterizedTypeConstant", "parameterizedtype.constant",  [
    DeclareAttrInterfaceMethods<SubElementAttrInterface, ["replaceImmediateSubElements"]>
  ]> {
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
  let storageClass = ConcreteTypeConstantAttr.storageClass;
  let genStorageClass = 0;

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
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def DTypeConstantAttr : ExprAttr<"DTypeConstant", "dtype.constant"> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to DType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "KGENDType":$dType, AttributeSelfTypeParameter<"">:$type);

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

def SymbolConstantAttr : ExprAttr<"SymbolConstant", "symbol.constant"> {
  let summary = "kgen.func or kgen.generator reference";
  let description = [{
    This is a value of !kgen.signature type, which refers to a func or
    generator.  This may optionally bind the input parameter values at time of
    formation - when this happens, the result type is non-parametric.
  }];

  let parameters = (ins "SymbolRefAttr":$symbolRef,
                        "ParamBindArrayAttr":$paramValues,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "Type":$type), [{
      auto *context = symbol.getContext();
      return get(context, symbol, ParamBindArrayAttr::get(context, {}), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ParamBindArrayAttr":$paramValues,
                                        "Type":$type), [{
      return get(symbol.getContext(), symbol, paramValues, type);
    }]>,
  ];

  let assemblyFormat =
     "`<` $symbolRef custom<OptionalParamBindSpec>($paramValues) `>`";

  let extraClassDeclaration = [{
    /// Get the symbol name.
    StringAttr getName() { return getSymbol().getAttr(); }

    /// Get the symbol constant as a flat reference.
    FlatSymbolRefAttr getSymbol() {
      return llvm::cast<FlatSymbolRefAttr>(getSymbolRef());
    }
  }];
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

def ConstraintAttr : KGENAttr<"Constraint", "constraint", [
    SubElementAttrInterface
  ]> {
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
    custom<ConstraintLoc>($loc) `]`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "TypedAttr":$expr, "StringAttr":$message, "Location":$loc), [{
      return $_get(expr.getContext(), expr, message, loc);
    }]>
  ];
}

/// An array of ConstraintAttr's.
def ConstraintArrayAttr
    : ArrayOfAttrsAttr<"ConstraintArray", "constraints", "ConstraintAttr">;

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

def TargetInfoAttr : ExprAttr<"TargetInfo", "target"> {
  let parameters = (ins StringRefParameter<"">:$triple,
                        StringRefParameter<"">:$cpu,
                        StringRefParameter<"">:$features,
                        "int64_t":$pointerSize,
                        AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = [{
    `<` $triple `,` $cpu `,` $features `,` $pointerSize `>`
  }];

  let extraClassDeclaration = [{
    /// Provides a stable hash function for use as part of a cache key.
    llvm::hash_code hash() const {
      return llvm::hash_combine(getTriple(), getCpu(), getFeatures(),
                                getPointerSize());
    }

    static TargetInfoAttr getForHost(::mlir::MLIRContext *ctx);
  }];
}

#endif // KGEN_KGENDIALECT_KGENATTRS_TD
