//===- KGENAttrs.td - Attributes for KGEN dialect -------------------------===//
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
  : Confined<TypedAttrInterface, [AttrConstraint<
      CPred<"KGEN::isValidParameterExpr($_self)">, "valid parameter expression"
    >]>;

def I1ParameterExprAttr
  : Confined<ParameterExprAttr, [AttrConstraint<
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
  : Confined<ParamDeclArrayAttr, [AttrConstraint<
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
  let parameters = (ins
    "StringAttr":$name,
    DefaultValuedParameter<"Type", "IndexType::get($_ctxt)">:$type,
    "TypedAttr":$value
  );

  let assemblyFormat =
    "custom<ParamName>($name) (`:` $type^)? `=` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type,
                                        "TypedAttr":$value), [{
      return get(name.getContext(), name, type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type,
                                        "TypedAttr":$value), [{
      return get(StringAttr::get(type.getContext(), name), type, value);
    }]>,
  ];
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

/// Fully associative variadic expressions.
def POC_Add : I32EnumAttrCase<"Add", 0, "add">;
def POC_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def POC_And : I32EnumAttrCase<"And", 2, "and">;
def POC_Or  : I32EnumAttrCase<"Or",  3, "or">;
def POC_Xor : I32EnumAttrCase<"Xor", 4, "xor">;

// Binary expressions.
def POC_Shl : I32EnumAttrCase<"Shl", 5, "shl">;
def POC_Shr : I32EnumAttrCase<"Shr", 6, "shr">;
def POC_Div : I32EnumAttrCase<"Div", 7, "div">;
def POC_Mod : I32EnumAttrCase<"Mod", 8, "mod">;

// Compares.
def POC_EQ : I32EnumAttrCase<"EQ", 9, "eq">;
def POC_LT : I32EnumAttrCase<"LT", 10, "lt">;
def POC_LE : I32EnumAttrCase<"LE", 11, "le">;

// Complex variadic operations.
def POC_IN : I32EnumAttrCase<"IN", 12, "in">;

// TODO: Can add compares, string concat, etc as needed.

/// This defines the Parameter Operator Code enum attribute.
def POCAttr  : I32EnumAttr<"POC", "Parameter Operator Code",
                           [POC_Add, POC_Mul, POC_And, POC_Or, POC_Xor,
                            POC_Shl, POC_Shr, POC_Div, POC_Mod,
                            POC_EQ, POC_LT, POC_LE, POC_IN]>;
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
    ), [{}], "TypedAttr">,

    // Build a parameter expression.  This automatically canonicalizes and
    // folds, so it may not necessarily return a ParamOperatorAttr.
    AttrBuilderWithInferredContext<(ins
      "POC":$opcode, "llvm::ArrayRef<TypedAttr>":$operands
    ), [{}], "TypedAttr">,

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
    /// Return (ne x, y) which is the same as (not x, y).
    static TypedAttr getNE(TypedAttr lhs, TypedAttr rhs) {
      return getNot(get(POC::EQ, lhs, rhs));
    }

    /// Return (not x) which is the same as (xor x, true).  The `operand` value
    /// must have type `i1`.
    static TypedAttr getNot(TypedAttr operand);
  }];
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
// ConstraintArrayAttr
//===----------------------------------------------------------------------===//

/// An array of attributes with bool type.
def ConstraintArrayAttr
  : TypedArrayAttrBase<I1ParameterExprAttr, "array of parameter constraints">;
