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

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

/// This attribute binds a concrete value to a named parameter, for example a
/// kgen.return uses it to bind returned parameters the the enclosing generators
/// parameter declarations.
def ParamBindAttr : AttrDef<KGENDialect, "ParamBind"> {
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
// ParamExprAttr
//===----------------------------------------------------------------------===//

// ParamExprAttr forms the core of the expression grammar for parameter
// expressions.  Its operators have been designed so that expression building
// force a deterministic canonical form on expression trees.  This ensures that
// `(x+1)*4` and `(x*4)+4` canonicalize into the same pointer value.
//
// For this reason, we should never have unary negate or subtract. `(x-y)` is
// modeled as `(x+-y)` => `(x + -1*y)`.  The ParamExprAttr class does provide
// convenience methods for these aggregate operations.
//
// The design of these operators assumes signful types like si64, and the fully
// associative operations are take arbitrary arity, also to make
// canonicalization simpler.

/// Parameter Expression Opcodes.
let cppNamespace = "M::KGEN" in {

/// Fully associative variadic expressions.
def PEO_Add : I32EnumAttrCase<"Add", 0, "add">;
def PEO_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def PEO_And : I32EnumAttrCase<"And", 2, "and">;
def PEO_Or  : I32EnumAttrCase<"Or",  3, "or">;
def PEO_Xor : I32EnumAttrCase<"Xor", 4, "xor">;

// Binary expressions.
def PEO_Shl : I32EnumAttrCase<"Shl", 5, "shl">;
def PEO_Shr : I32EnumAttrCase<"Shr", 6, "shr">;
def PEO_Div : I32EnumAttrCase<"Div", 7, "div">;
def PEO_Mod : I32EnumAttrCase<"Mod", 8, "mod">;

// TODO: Can add compares, string concat, etc as needed.

def PEOAttr  : I32EnumAttr<"PEO", "Parameter Expression Opcode",
                           [PEO_Add, PEO_Mul, PEO_And, PEO_Or, PEO_Xor,
                            PEO_Shl, PEO_Shr, PEO_Div, PEO_Mod]>;
}

def ParamExprAttr : AttrDef<KGENDialect, "ParamExpr"> {
  let summary = "Parameter expression";
  let parameters = (ins "PEO":$opcode,
                        ArrayRefParameter<"::mlir::Attribute">:$operands,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.expr";

  // Force all clients to go through our building logic so we can canonicalize
  // during building.
  let skipDefaultBuilders = 1;
 
  let extraClassDeclaration = [{
    /// Build a parameter expression.  This automatically canonicalizes and
    /// folds, so it may not necessarily return a ParamExprAttr.
    static Attribute get(PEO opcode, mlir::ArrayRef<mlir::Attribute> operands);

    /// Builder used by the generic parser.
    static Attribute get(MLIRContext *ctx, PEO opcode,
                         mlir::ArrayRef<mlir::Attribute> operands,
                         mlir::Type type);

    /// Build a binary parameter expression for convenience.
    static Attribute get(PEO opcode, mlir::Attribute lhs, mlir::Attribute rhs) {
      Attribute operands[] = { lhs, rhs };
      return get(opcode, operands);
    }
  }];

  let assemblyFormat = "`<` $opcode `,` $operands `>`";

 // let hasCustomAssemblyFormat = 1;
}
