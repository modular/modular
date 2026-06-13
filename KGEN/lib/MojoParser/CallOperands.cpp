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

/// Validate the operand list against the signature indicated by
/// pogListAttr, emitting an error with "getDiag" if invalid.
///
/// This populates "pogAssignment" with information about the mapping of
/// operands to POG entries.
LogicalResult CallOperands::assignToPogs(
    PogListAttr pogListAttr, bool isParameterList, PogAssignment &pogAssignment,
    llvm::function_ref<MojoInflightDiag &(llvm::SMLoc)> getDiag) const {

  // The specified operand is being passed to a non-variadic arg/param.  If it
  // is unpacked with `*arg` emit an error.
  auto checkPackSplat = [&](size_t opIdx) -> LogicalResult {
    if (values[opIdx].unpackStyle == ArgUnpackStyle::kPositional ||
        values[opIdx].unpackStyle == ArgUnpackStyle::kKeyword)
      return success();
    getDiag(values[opIdx].expr->getLoc())
        << "unpack is only supported when the callee accepts a "
           "variadic; to forward a runtime pack to a fixed-arity callee, "
           "route the call through a dispatcher whose argument is itself a "
           "variadic pack (e.g. "
           "`def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: "
           "def(*args: *Ts) thin](...): callee(*pack)`)"
        << values[opIdx].expr->getRange();
    return failure();
  };

  // Scan each pog and figure out which operands contribute to it. We know that
  // the POG list is validated, so positional arguments will precede keyword
  // arguments, but keyword arguments can be specified in call operands in any
  // order:   foo(kw=42, 7).
  SmallVector<size_t> skippedKW;

  size_t opIdx = 0, numOperands = values.size();
  for (auto [idx, pog] : llvm::enumerate(pogListAttr.getPogs())) {
    PassingKind passingKind = pog.getPassingKind();

    // Ignore implicit operands like out parameters: byref_result and error.
    if (passingKind == PassingKind::Implicit) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Unspecified);
      continue;
    }

    // Skip over any provided keyword operands when matching things up, we
    // handle them separately below.
    while (opIdx < numOperands && values[opIdx].keyword) {
      skippedKW.push_back(opIdx);
      ++opIdx;
    }

    // For positional variadics and packs, consume any positional operands.
    if (pog.isPosVarArg() || pog.isPack()) {
      // Note that the contents will be captured in the posVariadicIdxs list.
      // Note this captures individual values as well as unpacks.
      pogAssignment.posVariadicIdxs.push_back(PogAssignment::kPA_Variadic);
      while (opIdx != numOperands) {
        if (!values[opIdx].keyword)
          pogAssignment.posVariadicIdxs.push_back(opIdx);
        else
          skippedKW.push_back(opIdx);
        ++opIdx;
      }
      continue;
    }

    // KW variadics eat up any remaining keyword operands, and accept **kwargs.
    if (pog.isKwVarArg()) {
      pogAssignment.posVariadicIdxs.push_back(PogAssignment::kPA_Variadic);

      // Start with any skipped kw operands.
      pogAssignment.kwVariadicIdxs = std::move(skippedKW);

      // Then eat up any remaining keyword operands.
      for (size_t opIdx = 0; opIdx < numOperands; ++opIdx) {
        if (values[opIdx].keyword)
          pogAssignment.kwVariadicIdxs.push_back(opIdx);
        else // Unassigned positional operands are errors.
          break;
      }
      continue;
    }

    // If we have a non-kw value and non-kw POG, bind the operand.
    if (opIdx < numOperands && (passingKind == PassingKind::PosOrKw ||
                                passingKind == PassingKind::PosOnly)) {
      if (failed(checkPackSplat(opIdx)))
        return failure();

      pogAssignment.operandIdxs.push_back(opIdx);
      ++opIdx;
      continue;
    }

    // If this POG allows a keyword operand and we have one, bind it.
    if (passingKind != PassingKind::PosOnly &&
        passingKind != PassingKind::Implicit) {
      if (const OperandValue *operand = findKwArg(pog.getName())) {
        size_t operandIdx = operand - values.begin();
        pogAssignment.operandIdxs.push_back(operandIdx);
        if (failed(checkPackSplat(operandIdx)))
          return failure();

        auto it = std::find(skippedKW.begin(), skippedKW.end(), operandIdx);
        if (it != skippedKW.end())
          skippedKW.erase(it);
        continue;
      }
    }

    // Parameters can be inferred for many calls, return them inferred.
    if (isParameterList) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Unspecified);
      continue;
    }

    // If there is a default value use it.
    if (auto defaultVal = pog.getDefaultValue()) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Default);
      continue;
    }

    // Otherwise, this is missing.
    const char *kindStr = isParameterList ? "parameter" : "argument";
    getDiag(getExprLoc()) << "missing required " << kindStr << ": "
                          << pog.getName();
    return failure();
  }

  // Now that we assigned operands to all POG entries, make sure we don't have
  // any excess operands.

  // If there are no positional variadics, we can check for too many operands.
  if (opIdx != numOperands) {
    const char *kindStr = isParameterList ? "parameter" : "argument";
    getDiag(values[opIdx].expr->getLoc())
        << "unexpected " << kindStr << values[opIdx].expr->getRange();
    return failure();
  }

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
        auto &diag = getDiag(getExprLoc());
        const char *kindStr = isParameterList ? "parameter" : "argument";
        diag << "missing required keyword-only " << kindStr << ": " << name;
        return failure();
      }
      continue;
    }
    if (passingKind == PassingKind::PosOnly) {
      if (!name.empty() && findKwArg(name)) {
        auto &diag = getDiag(getExprLoc());
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
  for (auto [operandIdx, operand] : llvm::enumerate(values)) {
    // Scan through inferred names. These must be specified in order.
    while (inferredNameIter != inferredNames.end() &&
           *inferredNameIter != operand.keyword)
      ++inferredNameIter;

    // Found a matching explicitly-specified inferred param.
    if (inferredNameIter != inferredNames.end())
      continue;

    if (inferredNames.contains(operand.keyword)) {
      auto &diag = getDiag(getExprLoc());
      diag << "inferred parameter passed out of order: " << operand.keyword;
      return failure();
    }

    if (operand.keyword && !kwPassableNames.contains(operand.keyword)) {
      // If the function doesn't accept variadic kwargs, this is an error.
      if (!pogListAttr.hasKwVarArg()) {
        auto &diag = getDiag(getExprLoc());
        const char *argOrParam = isParameterList ? "parameter" : "argument";
        diag << "unknown keyword " << argOrParam << ": " << operand.keyword;
        return failure();
      }
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
        auto &diag = getDiag(getExprLoc());
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

    auto &diag = getDiag(getExprLoc());
    diag << "missing required positional argument: " << name;
    attachPackSplatNote(diag);
    return failure();
  }

  if (!isParameterList) { // Parameters can be inferred, don't error on missing.
    // If there are no positional variadics, we can check for too many operands.
    if (!hasVariadicOrPack && getNumPositional() > numPosMaximum) {
      auto &diag = getDiag(getExprLoc());
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
