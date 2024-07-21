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
// CallOperands
//===----------------------------------------------------------------------===//

/// This is an operand record, maintaining the IR repre that might
struct OperandValue : public ASTExprAnd<AnyValue> {
  // Null for positional arguments.
  StringAttr keyword;

  OperandValue(StringAttr keyword, ASTExprAnd<AnyValue> value)
      : ASTExprAnd<AnyValue>(std::move(value)), keyword(keyword) {}
};

using OperandValueList = SmallVector<OperandValue, 4>;

/// Struct that carries both positional and keyword operands for a call or
/// parameter binding. This does not own any values, only references pointers
/// to their containers.
class CallOperands {
public:
  /// Create call operands with positional and optional keyword arguments.
  CallOperands(ArrayRef<ASTExprAnd<AnyValue>> posOperands = {})
      : posOperands(posOperands) {}

  CallOperands(CallOperands &&) = default;
  explicit CallOperands(const CallOperands &) = default;
  CallOperands &operator=(CallOperands &&) = default;

  /// Return a keyword argument value if present, or null otherwise.
  std::optional<ASTExprAnd<AnyValue>> findKwArg(StringAttr argName) const {
    auto it = kwOperands.find(argName);
    if (it != kwOperands.end())
      return it->second;
    return std::nullopt;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const { return kwOperands.size(); }

  /// The values passed as positional operands.
  SmallVector<ASTExprAnd<AnyValue>, 4> posOperands;

  /// The values passed as keyword operands.
  llvm::MapVector<StringAttr, ASTExprAnd<AnyValue>,
                  SmallDenseMap<StringAttr, size_t>>
      kwOperands;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;

  bool empty() const { return posOperands.empty() && kwOperands.empty(); }

  //===--------------------------------------------------------------------===//
  // Manipulators
  //===--------------------------------------------------------------------===//

  /// Add a positional argument to the list.
  void add(ASTExprAnd<AnyValue> &&value) { posOperands.push_back(value); }

  /// Add positional arguments to the list.
  void add(ArrayRef<ASTExprAnd<AnyValue>> values) {
    posOperands.append(values.begin(), values.end());
  }

  /// Add a keyword argument, there can never be conflicts here because keyword
  /// argument conflicts should be checked in the parser before any semantic
  /// analysis is attempted.
  void add(StringAttr name, ASTExprAnd<AnyValue> value) {
    auto [_, addedNew] = kwOperands.try_emplace(name, std::move(value));
    assert(addedNew && "duplicate keywords should be detected at parse time, "
                       "before semantic analysis");
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
                          OperandValueList &variadicKwOperands,
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

raw_ostream &operator<<(raw_ostream &os, const CallOperands &value);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_OPERANDCONTAINER_H