//===- KGENAttrs.td - Attributes for KGEN dialect -------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

/// This is an attribute that is formed of valid nodes in a parameter
/// expression.
def ParameterExprAttr
  : Confined<AnyAttr, [AttrConstraint<
      CPred<"KGEN::isValidParameterExpr($_self)">, "valid parameter expression"
    >]>;

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

/// Declaration of a named parameter.
def ParamDeclAttr : AttrDef<KGENDialect, "ParamDecl"> {
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
  let mnemonic = "param.decl";

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
def ParamBindAttr : AttrDef<KGENDialect, "ParamBind",
              [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                           ["replaceImmediateSubAttribute"]>]> {
  let summary = "Bind a value to a parameter declaration.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.bind";
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
def ParamDeclRefAttr : AttrDef<KGENDialect, "ParamDeclRef"> {
  let summary = "A reference to a parameter value.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.decl.ref";

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

// TODO: Can add compares, string concat, etc as needed.

/// This defines the Parameter Operator Code enum attribute.
def POCAttr  : I32EnumAttr<"POC", "Parameter Operator Code",
                           [POC_Add, POC_Mul, POC_And, POC_Or, POC_Xor,
                            POC_Shl, POC_Shr, POC_Div, POC_Mod]>;
}

def ParamOperatorAttr : AttrDef<KGENDialect, "ParamOperator",
              [DeclareAttrInterfaceMethods<SubElementAttrInterface,
                                           ["replaceImmediateSubAttribute"]>]> {
  let summary = "Parameter expression";
  let parameters = (ins "POC":$opcode,
                        ArrayRefParameter<"::mlir::Attribute">:$operands,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.expr";

  // Force all clients to go through our building logic so we can canonicalize
  // during building.
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Build a parameter expression.  This automatically canonicalizes and
    /// folds, so it may not necessarily return a ParamOperatorAttr.
    static Attribute get(POC opcode, mlir::ArrayRef<mlir::Attribute> operands);

    /// Builder used by the generic parser.
    static Attribute get(MLIRContext *ctx, POC opcode,
                         mlir::ArrayRef<mlir::Attribute> operands,
                         mlir::Type type);

    /// Build a binary parameter expression for convenience.
    static Attribute get(POC opcode, mlir::Attribute lhs, mlir::Attribute rhs) {
      Attribute operands[] = { lhs, rhs };
      return get(opcode, operands);
    }
  }];

  let assemblyFormat = "`<` $opcode `,` $operands `>`";
  let genVerifyDecl = 1;  // Generate getChecked/verify.
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def DTypeConstantAttr : AttrDef<KGENDialect, "DTypeConstant"> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to
    TensorEltType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "::mlir::IntegerAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "dtype.constant";

  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return the TensorEltType for the value we contain.
    TensorEltType getTensorEltType();
  }];
}
