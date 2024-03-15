//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares diagnostic utilities used by call check and emission code.
//
//===----------------------------------------------------------------------===//

#ifndef MOJOPARSER_OPERANDDIAGNOSTICS_H
#define MOJOPARSER_OPERANDDIAGNOSTICS_H

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "KGEN/MojoParser/CallEmission.h"
#include "llvm/ADT/StringSet.h"

namespace M::KGEN::LIT {

/// Designates the kind of keyword-operand errors.
enum class KwDiagResult {
  kValid,
  kMissingKwOnly,
  kPosOnlyPassedByKw,
  kUnknownKeywords
};

/// Helper to diagnose common cases of candidate mismatch related to keyword
/// operands (unexpected kw-operands, pos-only arg/param provided by kw-operand,
/// missing kw-only arg/param). If the function accepts variadic keyword
/// args/params, this function also collects them.
template <typename OperandType>
static std::pair<KwDiagResult, SmallVector<StringAttr>> diagnoseKeywordOperands(
    PogsAttr pogsAttr, KeywordOperandContainer<OperandType> &variadicKwOperands,
    const OperandContainer<OperandType> &operands,
    bool allowMissingKwOnly = false) {
  // First, we collect any (named) pos-only args/params passed by keyword
  // operand, and missing kw-only args/params. We also collect all arg/param
  // names that might be specified by keyword.
  StringSet<> kwPassableNames;
  SmallVector<StringAttr> posOnlyPassedByKw;
  SmallVector<StringAttr> missingKwOnly;

  DefaultValueHandler defaultHandler(pogsAttr);
  for (auto [argIdx, name, passingKind] :
       llvm::enumerate(pogsAttr.getNames(), pogsAttr.getPassingKinds())) {
    if (passingKind == PassingKind::Implicit)
      continue;
    if (pogsAttr.isPack(argIdx) || pogsAttr.isVariadic(argIdx))
      continue; // Variadic/pack args cannot be specified by their keyword.
    if (passingKind == PassingKind::KwOnly &&
        !defaultHandler.getKwOnlyDefault(argIdx) && !operands.findKwArg(name)) {
      missingKwOnly.push_back(name);
      continue;
    }
    if (passingKind == PassingKind::PosOnly) {
      if (!name.empty() && operands.findKwArg(name))
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
  if (operands.hasKwOperands()) {
    for (auto [name, operand] : *operands.kwOperands)
      if (!kwPassableNames.contains(name))
        variadicKwOperands.try_emplace(name, operand);
  }

  // If the function doesn't accept variadic kwargs, this is an error.
  if (!pogsAttr.hasKwVariadics() && !variadicKwOperands.empty()) {
    SmallVector<StringAttr> unknownKwOperands;
    for (auto [name, _] : variadicKwOperands)
      unknownKwOperands.push_back(name);
    return {KwDiagResult::kUnknownKeywords, unknownKwOperands};
  }

  return {KwDiagResult::kValid, {}};
}

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_OPERANDDIAGNOSTICS_H
