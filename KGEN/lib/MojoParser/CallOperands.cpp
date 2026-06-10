//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains code pertaining to manipulation and diagnostics for
// operand/parameter list processing.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CallOperands.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/MojoDiags.h"
#include "MojoUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

StringRef LIT::stringifyCallSyntax(CallSyntax val) {
  switch (val) {
  case CallSyntax::kParamBindings:
    return "param_bindings";
  case CallSyntax::kDirectCall:
    return "direct_call";
  case CallSyntax::kIndirectCall:
    return "indirect_call";
  case CallSyntax::kMethodCall:
    return "method_call";
  case CallSyntax::kTypeCall:
    return "type_call";
  case CallSyntax::kOperator:
    return "operator";
  case CallSyntax::kReversedOperator:
    return "reversed_operator";
  case CallSyntax::kSubscript:
    return "subscript";
  case CallSyntax::kAttribute:
    return "attribute";
  case CallSyntax::kImplicitConvert:
    return "implicit_convert";
  case CallSyntax::kImplicitCopyCtor:
    return "implicit_copy";
  case CallSyntax::kImplicitMoveCtor:
    return "implicit_move";
  case CallSyntax::kDestructor:
    return "destructor";
  case CallSyntax::kTupleGetItem:
    return "tuple_get_item";
  case CallSyntax::kMethodCallSynthetic:
    return "method_call_synthetic";
  }
  llvm_unreachable("unknown CallSyntax");
  return "";
}

raw_ostream &LIT::operator<<(raw_ostream &os, CallSyntax val) {
  return os << stringifyCallSyntax(val);
}

//===----------------------------------------------------------------------===//
// CallOperands
//===----------------------------------------------------------------------===//

llvm::SMLoc CallOperands::getExprLoc() const { return callExpr->getLoc(); }

void CallOperands::dump() const { llvm::errs() << *this << '\n'; }

raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os,
                                      const CallOperands &operands) {
  os << "CallOperands{ " << operands.getNumPositional() << " pos args, "
     << operands.getNumKwOperands() << " kw args";
  if (operands.hasSelfOperand)
    os << " <HAS SELF OPERAND>";
  os << '\n';

  for (auto &operand : operands.values) {
    os << "  ";
    if (operand.keyword)
      os << operand.keyword.getValue() << ": ";
    os << operand.ir << "\n";
  }
  return os << '}';
}

/// Validation the operand list against the signature indicated by
/// pogListAttr, emitting an error with "getDiag" if invalid.
///
/// This collects variadic keyword args/params if the function allows them.
LogicalResult CallOperands::diagnoseOperands(
    PogListAttr pogListAttr, OperandValueList &variadicKwOperands,
    bool isParameterList,
    llvm::function_ref<MojoInflightDiag &()> getDiag) const {

  // First, we collect any (named) pos-only args/params passed by keyword
  // operand, and missing kw-only args/params. We also collect all arg/param
  // names that might be specified by keyword.
  llvm::SetVector<StringAttr> inferredNames;
  SmallPtrSet<StringAttr, 4> kwPassableNames;

  for (auto [argIdx, pogAttr] : llvm::enumerate(pogListAttr.getPogs())) {
    StringAttr name = pogAttr.getName();
    if (name.empty())
      continue;
    PassingKind passingKind = pogAttr.getPassingKind();
    if (passingKind == PassingKind::Inferred) {
      inferredNames.insert(name);
      continue;
    }
    // Implicit parameter cannot be passed by the user. They must be
    // inferred from values bound to parameters or arguments, so we don't have
    // to verify them here.  Inferred parameters can be bound by name.
    if (passingKind == PassingKind::Implicit)
      continue;
    if (pogListAttr.isAnyVarArg(argIdx))
      continue; // Variadic/pack args cannot be specified by their keyword.
    if (passingKind == PassingKind::KwOnly && !pogListAttr.getDefault(argIdx) &&
        !findKwArg(name)) {
      if (!isParameterList) { // KWOnly parameters may be inferred.
        auto &diag = getDiag();
        const char *kindStr = isParameterList ? "parameter" : "argument";
        diag << "missing required keyword-only " << kindStr << ": " << name;
        return failure();
      }
      continue;
    }
    if (passingKind == PassingKind::PosOnly) {
      if (!name.empty() && findKwArg(name)) {
        auto &diag = getDiag();
        const char *argOrParam = isParameterList ? "parameter" : "argument";
        diag << "positional-only " << argOrParam
             << " passed as keyword operand: " << name;
        return failure();
      }
      continue;
    }
    [[maybe_unused]] auto [_, addedNew] = kwPassableNames.insert(name);
    assert(addedNew && "duplicate argument/parameter name in signature");
  }

  // Collect all the keyword operands with unknown names.
  auto inferredNameIter = inferredNames.begin();
  for (auto &operand : values) {
    // Scan through inferred names. These must be specified in order.
    while (inferredNameIter != inferredNames.end() &&
           *inferredNameIter != operand.keyword)
      ++inferredNameIter;

    // Found a matching explicitly-specified inferred param.
    if (inferredNameIter != inferredNames.end())
      continue;

    if (inferredNames.contains(operand.keyword)) {
      auto &diag = getDiag();
      diag << "inferred parameter passed out of order: " << operand.keyword;
      return failure();
    }

    if (operand.keyword && !kwPassableNames.contains(operand.keyword)) {
      // If the function doesn't accept variadic kwargs, this is an error.
      if (!pogListAttr.hasKwVarArg()) {
        auto &diag = getDiag();
        const char *argOrParam = isParameterList ? "parameter" : "argument";
        diag << "unknown keyword " << argOrParam << ": " << operand.keyword;
        return failure();
      }

      // Otherwise remember it.
      variadicKwOperands.push_back(operand);
    }
  }

  // If any operand is a `*pack` splat and something didn't match, attach
  // a note explaining the gap and pointing at the working pattern.
  // `CallNode::emitIR` eagerly expands splats whose `VariadicPack`
  // element list is statically resolved; this note primarily helps the
  // case where the element list is still symbolic (e.g. `Ts.values` in
  // a generic body), but it's also useful when a resolved splat happens
  // to have the wrong element count.
  auto attachPackSplatNote = [&](MojoInflightDiag &diag) {
    for (const OperandValue &op : values) {
      if (op.unpackStyle != ArgUnpackStyle::kStar)
        continue;
      diag.attachNote(op.expr->getLoc())
          << "'*' splat is only supported when the callee accepts a "
             "variadic pack argument at this position; to forward a "
             "runtime pack to a fixed-arity callee, route the call through "
             "a dispatcher whose argument is itself a variadic pack (e.g. "
             "`def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: "
             "def(*args: *Ts) thin](...): callee(*pack)`)";
      return;
    }
  };

  size_t numOperands = values.size();
  size_t numPosMinimum = countNumInferredKinds(pogListAttr);
  size_t numPosMaximum = numPosMinimum + countNumPositional(pogListAttr);
  bool hasVariadicOrPack = false;

  size_t nextPosOperand = 0;

  // This loop is walking 'idx' in order of posListAttr, checking just the
  // positional arguments, not walking the operands list.
  for (size_t idx = numPosMinimum; idx != numPosMaximum; ++idx) {
    if (pogListAttr.isPosVarArg(idx) || pogListAttr.isPack(idx)) {
      // Positional variadics and packs don't require any operands. But we
      // remember this because it lifts the limit on the maximum number.
      hasVariadicOrPack = true;
      continue;
    }

    // Figure out the next positional operand.
    while (nextPosOperand < numOperands && values[nextPosOperand].keyword)
      ++nextPosOperand;

    // If we found a positional operand, check if it was also provided by
    // keyword.
    if (nextPosOperand < numOperands) {
      StringAttr name = pogListAttr.getName(idx);
      if (findKwArg(name)) {
        const char *argOrParam = isParameterList ? "parameter" : "argument";
        auto &diag = getDiag();
        diag << argOrParam
             << " passed both as positional and keyword operand: " << name;
        return failure();
      }
      ++nextPosOperand;
      continue;
    }

    // If we have a positional default, we're okay. We also don't need to check
    // for missing if the caller doesn't care about them.
    if (isParameterList || pogListAttr.getDefault(idx))
      continue;

    // If the arg/param was passed by keyword, we are okay.
    StringAttr name = pogListAttr.getName(idx);
    if (findKwArg(name))
      continue;

    // Otherwise, we have a missing positional arg/param.
    if (name.empty()) {
      // TODO: fix "arg" below
      name = StringAttr::get(
          name.getContext(),
          "(" + ("positional-only " + Twine("arg") + " #" + Twine(idx)).str() +
              ")");
    }

    auto &diag = getDiag();
    diag << "missing required positional argument: " << name;
    attachPackSplatNote(diag);
    return failure();
  }

  if (!isParameterList) { // Parameters can be inferred, don't error on missing.
    // If there are no positional variadics, we can check for too many operands.
    if (!hasVariadicOrPack && getNumPositional() > numPosMaximum) {
      auto &diag = getDiag();
      size_t numPosMaximum = countNumPositional(pogListAttr);
      size_t numPosOperands = getNumPositional();
      diag << "expected at most " << numPosMaximum << " positional argument"
           << plural(numPosMaximum) << ", got " << numPosOperands;
      attachPackSplatNote(diag);
      return failure();
    }
  }

  return success();
}
