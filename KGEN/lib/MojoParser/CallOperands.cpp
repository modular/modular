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
                                      const CallOperands &value) {
  os << "CallOperands{ " << value.posOperands.size() << " pos args, "
     << value.getNumKwOperands() << " kw args";
  if (value.hasSelfOperand)
    os << " <HAS SELF OPERAND>";
  os << '\n';

  for (auto operand : value.posOperands)
    os << "  " << operand.ir << "\n";

  if (!value.kwOperands.empty()) {
    os << "Keyword bindings:\n";
    for (auto [name, binding] : value.kwOperands)
      os << "  " << name.getValue() << ": " << binding.ir.getIfPValue() << "\n";
  }
  return os << '}';
}

/// Helper to diagnose common cases of candidate mismatch related to keyword
/// operands (unexpected kw-operands, pos-only arg/param provided by kw-operand,
/// missing kw-only arg/param). If the function accepts variadic keyword
/// args/params, this function also collects them.
std::pair<CallOperands::KwDiagResult, SmallVector<StringAttr>>
CallOperands::diagnoseKeywordOperands(
    PogListAttr pogListAttr, KeywordOperandContainer &variadicKwOperands,
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
  for (auto [name, operand] : kwOperands)
    if (!kwPassableNames.contains(name))
      variadicKwOperands.try_emplace(name, operand);

  // If the function doesn't accept variadic kwargs, this is an error.
  if (!pogListAttr.hasKwVariadics() && !variadicKwOperands.empty()) {
    SmallVector<StringAttr> unknownKwOperands;
    for (auto [name, _] : variadicKwOperands)
      unknownKwOperands.push_back(name);
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

  size_t numPosOperands = posOperands.size();
  size_t numPosMaximum = countNumPositional(pogListAttr);
  bool hasVariadicOrPack = false;

  DefaultValueHandler defaultHandler(pogListAttr);
  for (size_t idx = 0; idx != numPosMaximum; ++idx) {
    if (pogListAttr.isPosVariadic(idx) || pogListAttr.isPack(idx)) {
      // Positional variadics and packs don't require any operands. But we
      // remember this because it lifts the limit on the maximum number.
      hasVariadicOrPack = true;
      continue;
    }

    // If we found a positional operand, check if it was also provided by
    // keyword.
    if (idx < numPosOperands) {
      StringAttr name = pogListAttr.getName(idx);
      if (findKwArg(name))
        byPosAndKw.push_back(name);
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
    if (!hasVariadicOrPack && numPosOperands > numPosMaximum)
      return {PosDiagResult::kTooManyPos, {}};

    if (!missingPosNames.empty())
      return {PosDiagResult::kMissingPos, missingPosNames};
  }

  return {PosDiagResult::kValid, {}};
}