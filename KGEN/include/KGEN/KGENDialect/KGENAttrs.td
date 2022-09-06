//===- KGENAttrs.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

/// This is an attribute that is formed of valid nodes in a parameter
/// expression.
def ParameterExprAttr
  : ConfinedAttr<TypedAttrInterface, [AttrConstraint<
      CPred<"KGEN::isValidParameterExpr($_self)">, "valid parameter expression"
    >]>;

def I1ParameterExprAttr
  : ConfinedAttr<ParameterExprAttr, [AttrConstraint<
      CPred<"$_self.cast<TypedAttr>().getType().isInteger(1)">,
            "i1 parameter expression"
    >]>;

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
def ParamDeclAttr : KGENAttr<"ParamDecl", "param.decl"> {
  let summary = "Parameter declaration";
  let description = [{
    This is a declaration of a parameter in the meta-programming domain for
    kernel generators and related infrastructure.  These are typically owned by
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
      "return get(StringAttr::get(type.getContext(), name), type);">,
    AttrBuilderWithInferredContext<(ins "StringAttr":$name), [{
      return get(name, IndexType::get(name.getContext()));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ArrayOfAttrsAttr
//===----------------------------------------------------------------------===//

/// An array attribute of another attribute.
class ArrayOfAttrsAttr<string attrName, string attrMnemonic, string eltName>
    : KGENAttr<attrName, attrMnemonic,
              [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let parameters = (ins ArrayRefParameter<eltName>:$value);
  let assemblyFormat = "`[` $value `]`";

  let returnType = "ArrayRef<" # eltName # ">";
  let constBuilderCall = "$_builder.getAttr<" # attrName # "Attr>($0)";
  let convertFromStorage = "$_self.getValue()";

  let extraClassDeclaration = [{
    auto begin() const { return getValue().begin(); }
    auto end() const { return getValue().end(); }
    bool empty() const { return getValue().empty(); }
    size_t size() const { return getValue().size(); }
    auto &front() const { return getValue().front(); }
  }];
}

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
// ParamBindAttr
//===----------------------------------------------------------------------===//

/// This attribute binds a concrete value to a named parameter, for example a
/// kgen.return uses it to bind returned parameters the the enclosing generators
/// parameter declarations.
def ParamBindAttr : KGENAttr<"ParamBind", "param.bind",
              [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                           ["replaceImmediateSubAttribute"]>]> {
  let summary = "Bind a value to a parameter declaration.";
  let parameters = (ins "StringAttr":$name, "TypedAttr":$value);

  let assemblyFormat = "custom<ParamName>($name) `=` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "TypedAttr":$value),
    [{
      return get(name.getContext(), name, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "TypedAttr":$value),
    [{
      return get(StringAttr::get(value.getContext(), name), value);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Return the type of the value being bound.
    Type getType() { return getValue().getType(); }
  }];
}

/// An array of ParamBindAttr's.
def ParamBindArrayAttr
    : ArrayOfAttrsAttr<"ParamBindArray", "param.binds", "ParamBindAttr">;

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

  let hasCustomAssemblyFormat = 1;
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

/// Parameter Expression Opcodes.
let cppNamespace = "M::KGEN" in {

/// This defines the Parameter Operator Code enum attribute.
def POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
  /// Fully associative variadic expressions.
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Mul", 1, "mul">,
  I32EnumAttrCase<"And", 2, "and">,
  I32EnumAttrCase<"Or",  3, "or">,
  I32EnumAttrCase<"Xor", 4, "xor">,

  // Binary expressions.
  I32EnumAttrCase<"Shl", 5, "shl">,
  I32EnumAttrCase<"Shr", 6, "shr">,
  I32EnumAttrCase<"Div", 7, "div">,
  I32EnumAttrCase<"Mod", 8, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 9, "eq">,
  I32EnumAttrCase<"LT", 10, "lt">,
  I32EnumAttrCase<"LE", 11, "le">,

  // Complex variadic operations.
  I32EnumAttrCase<"IN", 12, "in">,

  // TODO: Can add string concat, etc as needed.
]>;
}

def ParamOperatorAttr : ExprAttr<"ParamOperator", "param.expr",
              [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                           ["replaceImmediateSubAttribute"]>]> {
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
  : ExprAttr<"ConcreteTypeConstant", "concretetype.constant"> {
  let summary = "Constant !kgen.type MLIR type without parameter references";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute referrs to a type.  The difference is
    that it is a `TypedAttr` so it can be a parameter expression, and has
    'kgen.type' itself.

    The bound type must not have type parameters.
  }];

  let parameters = (ins "Type":$value, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $value `>`";
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$value)>,
    // This builder is required by ODS.
    AttrBuilder<(ins "mlir::Type":$value, "mlir::Type":$attrType), [{
      assert(attrType.isa<KGEN::MLIRTypeType>()); return get(value);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// ParameterizedTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This represents a parameter whose value is an MLIR type.  It is like
/// `TypeAttr` (an attribute referring to a type) but is a `TypedAttr` that
/// refers to an MLIR type and has 'kgen.type' itself.
def ParameterizedTypeConstantAttr
  : ExprAttr<"ParameterizedTypeConstant", "parameterizedtype.constant"> {
  let summary = "Constant !kgen.type MLIR type with parameter references";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute referrs to a type.  The difference is
    that it is a `TypedAttr` so it can be a parameter expression, and has
    'kgen.type' itself.

    The bound type is known to have type parameters.
  }];

  let parameters = (ins "Type":$value, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $value `>`";
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$value), "", "TypedAttr">,
    // This builder is required by ODS.
    AttrBuilder<(ins "mlir::Type":$value, "mlir::Type":$attrType), [{
      assert(attrType.isa<KGEN::MLIRTypeType>()); return get(value);
    }], "TypedAttr">,
  ];
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
  let parameters = (ins "DType":$dType, AttributeSelfTypeParameter<"">:$type);

  let builders = [AttrBuilder<(ins "DType":$dtype)>];

  let assemblyFormat = "`<` $dType `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Checks if the DType is compatible with the MLIR type.
    /// Types are compatible if the DType and the MLIR types are the same. I.e.
    /// The kind (integral or floating) is the same as well as the bitwidth.
    bool isCompatibleWith(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

def ConstraintAttr : KGENAttr<"Constraint", "constraint",
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "kgen constraint";
  let description = [{
    A constraint attribute represents a constraint on a kernel generator or
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
// HistogramAttr
//===----------------------------------------------------------------------===//

/// Custom histogram parameter - just a vector of std::pair.
def HistogramParam
  : ArrayRefParameter<"std::pair<size_t, uint64_t>",
                      "Histogram parameter for Attributes"> {
  let printer = "printHistogram($_printer, $_self)";
  let parser = "parseHistogram($_parser)";
}

/// Histogram attribute that applies per-unknown-size. The bucket label is the
/// dimension size in this usage.
def HistogramAttr : KGENAttr<"Histogram", "eval.histogram"> {
  let summary = "histogram of (bucket label, weight)";
  let description = [{
    This attribute describes a histogram represented as a list of
    (bucket label, weight). The list should be read as:

    ```mlir
    #kgen.eval.histogram<[
      (3, 256), ///< The first bucket is of label "3" and has 256 occurrences.
      (1, 10)   ///< The second bucket is of label "1" and has 10 occurrences.
    ]>
    ```
  }];

  let parameters = (ins HistogramParam:$buckets);

  let assemblyFormat = "`<` $buckets `>`";
}

//===----------------------------------------------------------------------===//
// InputGenKindAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def InputGenKindAttr : I32EnumAttr<"InputGenKind", "Input Generation Kind", [
  I32EnumAttrCase<"Zeros", 0, "zeros">,
  I32EnumAttrCase<"Ones", 1, "ones">,
  I32EnumAttrCase<"Random", 2, "random">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// EvalConfigurationAttr
//===----------------------------------------------------------------------===//

/// Eval configuration for a single argument. Binds size integers in order, so
/// if there are 2 unknown sizes, the first size will bind to the first
/// histogram, the second will bind to the second histogram, and so on.
def EvalArgConfigurationAttr
  : KGENAttr<"EvalArgConfiguration", "eval.arg.configuration"> {
  let summary = "evaluation configuration for a single argument";
  let parameters = (ins "InputGenKind":$genKind,
                        ArrayRefParameter<"HistogramAttr">:$sizeHistograms);

  let assemblyFormat = "`<` $genKind `,` $sizeHistograms `>`";
}

/// Eval configuration for a whole generator interface. The user should specify
/// single EvalArgConfiguration for each argument on the interface that has a
/// dynamic/unbound shape.
def EvalConfigurationAttr
  : ArrayOfAttrsAttr<"EvalConfiguration", "eval.configuration",
                     "EvalArgConfigurationAttr"> {
  let summary = "how to evaluate this generator interface";
  let description = [{
    This attribute provides a list of configuration attributes for each
    argument to the attached operation (generally
    `kgen.generator.interface`). This attribute is used by the elaborator
    to determine which implementations of a given interface are 'good'.

    The user should provide for each argument:
    ```mlir
    kgen.generator.interface(%arg0 : !meta.buffer<?, f32>, ...) attributes {
    evalCfg = [
      #kgen.eval.arg.configuration<
        zeros, ///< What kind of inputs to generate.
        #kgen.eval.histogram<[(3, 256), (1, 10)]> ///< The sizes/weights to use
                                                  ///< for the `?` size
      >,
      ...
    ]}
    ```
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "ArrayRef<EvalArgConfigurationAttr>":$args), [{
      return $_get(args.front().getContext(), args);
    }]>
  ];
}
