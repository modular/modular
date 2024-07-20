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
                   const KeywordOperandContainer *kwOperands = nullptr)
      : posOperands(posOperands), kwOperands(kwOperands) {}

  /// Create call operands with positional arguments given a value implicitly
  /// convertible to `ArrayRef`.
  template <
      typename OperandsT,
      typename = std::enable_if_t<
          !std::is_same_v<OperandsT, ArrayRef<ASTExprAnd<AnyValue>>> &&
          std::is_convertible_v<OperandsT, ArrayRef<ASTExprAnd<AnyValue>>>>>
  OperandContainer(OperandsT &&posOperands,
                   const KeywordOperandContainer *kwOperands = nullptr)
      : OperandContainer(ArrayRef<ASTExprAnd<AnyValue>>(
                             std::forward<OperandsT>(posOperands)),
                         kwOperands) {}

  /// Return a keyword argument value if present, or null otherwise.
  std::optional<ASTExprAnd<AnyValue>> findKwArg(StringAttr argName) const {
    if (hasKwOperands())
      if (auto it = kwOperands->find(argName); it != kwOperands->end())
        return it->second;
    return std::nullopt;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const {
    return kwOperands ? kwOperands->size() : 0;
  }

  /// Return if there are any keyword operands specified.
  bool hasKwOperands() const { return getNumKwOperands(); }

  /// The values passed as positional operands.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands;

  /// The values passed as keyword operands.
  const KeywordOperandContainer *kwOperands;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;

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