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

/// An attribute in the KGEN dialect.
class LITAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LITDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// VoidAttr
//===----------------------------------------------------------------------===//

def VoidAttr : LITAttr<"Void", "void", [TypedAttrInterface]> {
  let summary = "A void value";
  let description = [{
    This attribute is used as a marker for returns from calls and other
    expressions that do not produce a value.
  }];

  let parameters = (ins "Type":$type);
  let assemblyFormat = "$type";
}

#endif // KGEN_LITDIALECT_LITATTRS_TD
