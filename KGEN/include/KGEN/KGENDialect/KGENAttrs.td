//===- KGENAttrs.td - Attributes for KGEN dialect -------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"

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
                        "::mlir::Type":$type);
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
// ParamBindAttr
//===----------------------------------------------------------------------===//

/// This attribute binds a concrete value to a named parameter, for example a
/// kgen.return uses it to bind returned parameters the the enclosing generators
/// parameter declarations.
def ParamBindAttr : AttrDef<KGENDialect, "ParamBind"> {
  let summary = "Bind a value to a parameter declaration.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::Type":$type,
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
