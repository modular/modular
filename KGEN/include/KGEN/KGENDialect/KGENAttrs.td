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

class KGENAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<KGENDialect, attrName, [TypedAttrInterface] # traits> {
  let mnemonic = attrMnemonic;
}

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
  /// asmsyntax defaults to si64.
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                         "::mlir::Type":$type),
      "return $_get(type.getContext(), name, type);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                         "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name), [{
      return get(name, IntegerType::get(name.getContext(), 64,
                                        IntegerType::Signed));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ParamDeclArrayAttr
//===----------------------------------------------------------------------===//

/// An array of ParamDeclAttr's typically used on a kgen.generator and other
/// ops that define parameters.
def ParamDeclArrayAttr
  : TypedArrayAttrBase<ParamDeclAttr, "parameter array attribute">;

/// An array of ParamDeclAttr of size 1.
def ParamDeclSize1ArrayAttr
  : TypedArrayAttrBase<ParamDeclAttr, "single parameter array attribute"> {
  let predicate = And<[
    ParamDeclArrayAttr.predicate, CPred<"$_self.cast<ArrayAttr>().size() == 1">
  ]>;
}

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
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type,
                        "::mlir::Attribute":$value);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Type":$type,
                                        "::mlir::Attribute":$value), [{
      return get(name.getContext(), name, type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "::llvm::StringRef":$name,
                                        "::mlir::Type":$type,
                                        "::mlir::Attribute":$value), [{
      return get(StringAttr::get(type.getContext(), name), type, value);
    }]>,
  ];
}

/// An array of ParamBindAttr's.
def ParamBindArrayAttr
  : TypedArrayAttrBase<ParamBindAttr, "parameter binding array attribute">;

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// This attribute models a reference to a named parameter.  The type of the
/// ParamDeclRefAttr must always be the same as the type of the parameter being
/// referenced.
def ParamDeclRefAttr : KGENAttr<"ParamDeclRef", "param.decl.ref"> {
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
// TODO: Need to determine canonical format for inverting a bool, e.g. multiply
// by -1?  We don't need both EQ and NE for example, we should be able to just
// have EQ and LT and be good.
def POC_EQ : I32EnumAttrCase<"EQ", 9, "eq">;
def POC_EQ_DTYPE : I32EnumAttrCase<"EQ_DTYPE", 10, "eq_dtype">;

// TODO: Can add compares, string concat, etc as needed.

/// This defines the Parameter Operator Code enum attribute.
def POCAttr  : I32EnumAttr<"POC", "Parameter Operator Code",
                           [POC_Add, POC_Mul, POC_And, POC_Or, POC_Xor,
                            POC_Shl, POC_Shr, POC_Div, POC_Mod,
                            POC_EQ, POC_EQ_DTYPE]>;
}

def ParamOperatorAttr : KGENAttr<"ParamOperator", "param.expr",
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
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def DTypeConstantAttr : KGENAttr<"DTypeConstant", "dtype.constant"> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to
    DType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "::mlir::IntegerAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return the DType for the value we contain.
    DType getDType();
  }];
}

//===----------------------------------------------------------------------===//
// ConstraintArrayAttr
//===----------------------------------------------------------------------===//

/// An array of attributes with bool type.
def ConstraintArrayAttr
  : TypedArrayAttrBase<I1ParameterExprAttr, "array of parameter constraints">;
