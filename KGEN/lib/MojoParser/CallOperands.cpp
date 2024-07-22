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
#include "MojoUtils.h"

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace M;
using namespace KGEN;
using namespace LIT;

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

/// Helper to diagnose common cases of candidate mismatch related to keyword
/// operands (unexpected kw-operands, pos-only arg/param provided by kw-operand,
/// missing kw-only arg/param). If the function accepts variadic keyword
/// args/params, this function also collects them.
std::pair<CallOperands::KwDiagResult, SmallVector<StringAttr>>
CallOperands::diagnoseKeywordOperands(PogListAttr pogListAttr,
                                      OperandValueList &variadicKwOperands,
                                      bool allowMissingKwOnly) const {
  // First, we collect any (named) pos-only args/params passed by keyword
  // operand, and missing kw-only args/params. We also collect all arg/param
  // names that might be specified by keyword.
  SmallPtrSet<StringAttr, 4> kwPassableNames;
  SmallVector<StringAttr> posOnlyPassedByKw;
  SmallVector<StringAttr> missingKwOnly;

  DefaultValueHandler defaultHandler(pogListAttr);
  for (auto [argIdx, pogAttr] : llvm::enumerate(pogListAttr.getPogs())) {
    StringAttr name = pogAttr.getName();
    PassingKind passingKind = pogAttr.getPassingKind();
    if (passingKind == PassingKind::Implicit)
      continue;
    if (pogListAttr.isPack(argIdx) || pogListAttr.isVariadic(argIdx))
      continue; // Variadic/pack args cannot be specified by their keyword.
    if (passingKind == PassingKind::KwOnly &&
        !defaultHandler.getKwOnlyDefault(argIdx) && !findKwArg(name)) {
      if (!allowMissingKwOnly)
        missingKwOnly.push_back(name);
      continue;
    }
    if (passingKind == PassingKind::PosOnly) {
      if (!name.empty() && findKwArg(name))
        posOnlyPassedByKw.push_back(name);
      continue;
    }
    auto [_, addedNew] = kwPassableNames.insert(name);
    assert(addedNew && "duplicate argument/parameter name in signature");
  }

  if (!allowMissingKwOnly && !missingKwOnly.empty())
    return {KwDiagResult::kMissingKwOnly, std::move(missingKwOnly)};
  if (!posOnlyPassedByKw.empty())
    return {KwDiagResult::kPosOnlyPassedByKw, std::move(posOnlyPassedByKw)};

  // Collect all the keyword operands with unknown names.
  for (auto &operand : values)
    if (operand.keyword && !kwPassableNames.contains(operand.keyword))
      variadicKwOperands.push_back(operand);

  // If the function doesn't accept variadic kwargs, this is an error.
  if (!pogListAttr.hasKwVariadics() && !variadicKwOperands.empty()) {
    SmallVector<StringAttr> unknownKwOperands;
    for (auto &operand : variadicKwOperands)
      unknownKwOperands.push_back(operand.keyword);
    return {KwDiagResult::kUnknownKeywords, unknownKwOperands};
  }

  return {KwDiagResult::kValid, {}};
}

/// Helper to diagnose common cases of candidate mismatch related to positional
/// arguments/parameter (too many positionals, missing positionals,
/// argument/parameter specified both by positional and keyword operands).
std::pair<CallOperands::PosDiagResult, SmallVector<StringAttr>>
CallOperands::diagnosePosOperands(PogListAttr pogListAttr,
                                  bool allowCountMismatch) const {
  SmallVector<StringAttr> missingPosNames;
  SmallVector<StringAttr> byPosAndKw;

  size_t numOperands = values.size();
  size_t numPosMaximum = countNumPositional(pogListAttr);
  bool hasVariadicOrPack = false;

  size_t nextPosOperand = 0;

  DefaultValueHandler defaultHandler(pogListAttr);

  // This loop is walking 'idx' in order of posListAttr, checking just the
  // positional arguments, not walking the operands list.
  for (size_t idx = 0; idx != numPosMaximum; ++idx) {
    if (pogListAttr.isPosVariadic(idx) || pogListAttr.isPack(idx)) {
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
      if (findKwArg(name))
        byPosAndKw.push_back(name);
      ++nextPosOperand;
      continue;
    }

    // If we have a positional default, we're okay. We also don't need to check
    // for missing if the caller doesn't care about them.
    if (allowCountMismatch || defaultHandler.getPosDefault(idx))
      continue;

    // If the arg/param was passed by keyword, we are okay.
    StringAttr name = pogListAttr.getName(idx);
    if (findKwArg(name))
      continue;

    // Otherwise, we have a missing positional arg/param.
    if (name.empty()) {
      // TODO: fix "arg" below
      name = StringAttr::get(name.getContext(),
                             "(" + nameForPosOnly(idx, "arg") + ")");
    }
    missingPosNames.push_back(name);
  }

  if (!byPosAndKw.empty())
    return {PosDiagResult::kByPosAndKw, byPosAndKw};

  if (!allowCountMismatch) {
    // If there are no positional variadics, we can check for too many operands.
    if (!hasVariadicOrPack && getNumPositional() > numPosMaximum)
      return {PosDiagResult::kTooManyPos, {}};

    if (!missingPosNames.empty())
      return {PosDiagResult::kMissingPos, missingPosNames};
  }

  return {PosDiagResult::kValid, {}};
}