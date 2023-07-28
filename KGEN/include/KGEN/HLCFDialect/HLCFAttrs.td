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

//===----------------------------------------------------------------------===//
// ForLoopBoundCmpPredicate
//===----------------------------------------------------------------------===//

def HLCF_ForLoopBoundCmpPredicate:
    HLCF_Enum<"ForLoopBoundCmpPredicate",
              "loop unroll bound compare predicate type", [
                           HLCF_EnumCase<"SGTRHS", 0, "sgtrhs">,
                           HLCF_EnumCase<"SGTLHS", 1, "sgtlhs">,
                           HLCF_EnumCase<"SLTRHS", 2, "sltrhs">,
                           HLCF_EnumCase<"SLTLHS", 3, "sltlhs">]>;

def HLCF_ForLoopBoundCmpPredicateAttr
  : EnumAttr<HLCFDialect,
             HLCF_ForLoopBoundCmpPredicate,
             "for_loop_bound_cmp_predicate">;

//===----------------------------------------------------------------------===//
// ForLoopIndVarCompute
//===----------------------------------------------------------------------===//

def HLCF_ForLoopIndVarCompute:
    HLCF_Enum<"ForLoopIndVarCompute",
              "loop unroll induction variable compute type", [
                           HLCF_EnumCase<"ADD", 0, "add">,
                           HLCF_EnumCase<"SUB", 1, "sub">]>;

def HLCF_ForLoopIndVarComputeAttr
  : EnumAttr<HLCFDialect,
             HLCF_ForLoopIndVarCompute,
             "for_loop_ind_var_compute">;

#endif // HLCFATTRS_TD
