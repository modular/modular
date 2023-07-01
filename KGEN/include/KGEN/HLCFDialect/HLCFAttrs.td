//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef HLCFATTRS_TD
#define HLCFATTRS_TD

include "KGEN/HLCFDialect/HLCFDialect.td"
include "mlir/IR/EnumAttr.td"

class HLCF_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::HLCF";
  let genSpecializedAttr = 0;
}

class HLCF_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

class HLCF_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HLCFDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LoopUnrollFactor
//===----------------------------------------------------------------------===//

def HLCF_LoopUnrollFull: HLCF_Enum<"LoopUnrollFull", "loop unroll full", [
                           HLCF_EnumCase<"Full", 0, "full">,
                           HLCF_EnumCase<"None", 1, "none">]>;

def HLCF_LoopUnrollFullAttr
  : EnumAttr<HLCFDialect, HLCF_LoopUnrollFull, "loop_unroll_full">;

#endif // HLCFATTRS_TD
