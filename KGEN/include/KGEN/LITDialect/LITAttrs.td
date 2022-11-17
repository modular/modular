//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the LITDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_TD
#define KGEN_LITDIALECT_LITATTRS_TD

include "KGEN/LITDialect/LITDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

/// An attribute in the LIT dialect.
class LITAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LITDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

def UnboundMLIROperationAttr
  : LITAttr<"UnboundMLIROperation", "unbound_mlir_operation",
            [TypedAttrInterface]> {
  let summary = "An unbound mlir operation";
  let description = [{
    This attribute is used by the Lit parser as a reference to an MLIR
    operation by name: it may have bound attributes as well.  When applied to
    value operands, it materializes into a specific MLIR operation instance
  }];

  // TODO: Support bound attributes.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "StringAttr":$name);
  let assemblyFormat = "$name $type";
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

def NoneAttr : LITAttr<"None", "none", [TypedAttrInterface]> {
  let summary = "A None value";
  let description = [{
    This attribute represents None values.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
}

#endif // KGEN_LITDIALECT_LITATTRS_TD
