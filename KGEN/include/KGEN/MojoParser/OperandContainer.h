//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_OPERANDCONTAINER_H
#define KGEN_MOJOPARSER_OPERANDCONTAINER_H

#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/TypeCheckScopeInfo.h"
#include "llvm/ADT/MapVector.h"

namespace M::KGEN::LIT {
class PogListAttr;

//===----------------------------------------------------------------------===//
// OperandContainer
//===----------------------------------------------------------------------===//

/// A shorthand to make keyword operand handling more readable.
using KeywordOperandContainer =
    llvm::MapVector<StringAttr, ASTExprAnd<AnyValue>,
                    SmallDenseMap<StringAttr, size_t>>;

/// Struct that carries both positional and keyword operands for a call or
/// parameter binding. This does not own any values, only references pointers
/// to their containers.
class OperandContainer {
public:
  /// Create call operands with positional and optional keyword arguments.
  OperandContainer(ArrayRef<ASTExprAnd<AnyValue>> posOperands = {},
                   KeywordOperandContainer &&kwOperands = {})
      : posOperands(posOperands), kwOperands(std::move(kwOperands)) {}

  /// Return a keyword argument value if present, or null otherwise.
  std::optional<ASTExprAnd<AnyValue>> findKwArg(StringAttr argName) const {
    if (auto it = kwOperands.find(argName); it != kwOperands.end())
      return it->second;
    return std::nullopt;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const { return kwOperands.size(); }

  /// The values passed as positional operands.
  SmallVector<ASTExprAnd<AnyValue>, 4> posOperands;

  /// The values passed as keyword operands.
  KeywordOperandContainer kwOperands;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;

  //===--------------------------------------------------------------------===//
  // Manipulators
  //===--------------------------------------------------------------------===//

  /// Add a positional argument to the list.
  void add(ASTExprAnd<AnyValue> &&value) { posOperands.push_back(value); }

  /// Add positional arguments to the list.
  void add(ArrayRef<ASTExprAnd<AnyValue>> values) {
    posOperands.append(values.begin(), values.end());
  }

  /// Add a keyword argument.  This returns true if there was a conflict.
  [[nodiscard]] bool add(StringAttr name, ASTExprAnd<AnyValue> value) {
    auto [_, addedNew] = kwOperands.try_emplace(name, std::move(value));
    return !addedNew;
  }

  /// This adds a "self" argument to the start of the positional argument list.
  void addSelf(ASTExprAnd<AnyValue> value) {
    assert(!hasSelfOperand && "Cannot add a self when one is already present");
    posOperands.insert(posOperands.begin(), value);
    hasSelfOperand = true;
  }

  //===--------------------------------------------------------------------===//
  // Diagnostic helpers.
  //===--------------------------------------------------------------------===//

  /// Designates the kind of keyword-operand errors.
  enum class KwDiagResult {
    kValid,
    kMissingKwOnly,
    kPosOnlyPassedByKw,
    kUnknownKeywords
  };

  /// Helper to diagnose common cases of candidate mismatch related to keyword
  /// operands (unexpected kw-operands, pos-only arg/param provided by
  /// kw-operand, missing kw-only arg/param). If the function accepts variadic
  /// keyword args/params, this function also collects them.
  std::pair<KwDiagResult, SmallVector<StringAttr>>
  diagnoseKeywordOperands(PogListAttr pogListAttr,
                          KeywordOperandContainer &variadicKwOperands,
                          bool allowMissingKwOnly = false) const;

  /// Designates the kind of positional operand errors.
  enum class PosDiagResult { kValid, kMissingPos, kTooManyPos, kByPosAndKw };

  /// Helper to diagnose common cases of candidate mismatch related to
  /// positional arguments/parameter (too many positionals, missing positionals,
  /// argument/parameter specified both by positional and keyword operands).
  std::pair<PosDiagResult, SmallVector<StringAttr>>
  diagnosePosOperands(PogListAttr pogListAttr,
                      bool allowCountMismatch = false) const;
};

raw_ostream &operator<<(raw_ostream &os, const OperandContainer &value);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_OPERANDCONTAINER_H