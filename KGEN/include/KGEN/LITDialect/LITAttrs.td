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
// NoneAttr
//===----------------------------------------------------------------------===//

def NoneAttr : LITAttr<"None", "none", [TypedAttrInterface]> {
  let summary = "A None value";
  let description = [{
    This attribute represents None values.
  }];

  let parameters = (ins "Type":$type);
  let assemblyFormat = "$type";
}

#endif // KGEN_LITDIALECT_LITATTRS_TD
