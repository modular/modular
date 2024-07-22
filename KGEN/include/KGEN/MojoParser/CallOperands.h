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
  /// Initialize with some positional arguments.
  CallOperands(ArrayRef<ASTExprAnd<AnyValue>> posOperands) {
    for (const auto &operand : posOperands)
      values.emplace_back(StringAttr(), operand);
  }

  CallOperands() = default;
  CallOperands(CallOperands &&) = default;
  explicit CallOperands(const CallOperands &) = default;
  CallOperands &operator=(CallOperands &&) = default;

  /// Return a keyword argument value if present, or null otherwise.
  const OperandValue *findKwArg(StringAttr keyword) const {
    assert(keyword && "cannot look up null keyword");
    for (auto &elt : values) {
      if (elt.keyword == keyword)
        return &elt;
    }
    return nullptr;
  }

  /// Return the number of positional operands.
  size_t getNumPositional() const {
    size_t result = 0;
    for (auto &value : values)
      if (!value.keyword)
        ++result;
    return result;
  }

  /// Return the number of keyword operands.
  size_t getNumKwOperands() const { return values.size() - getNumPositional(); }

  /// The values passed in.  The keyword field will be null for positional
  /// arguments and present for keyword operands.
  OperandValueList values;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;

  //===--------------------------------------------------------------------===//
  // Element Accessors
  //===--------------------------------------------------------------------===//

  bool empty() const { return values.empty(); }
  size_t size() const { return values.size(); }

  const OperandValue &operator[](size_t index) const { return values[index]; }
  OperandValue &operator[](size_t index) { return values[index]; }

  //===--------------------------------------------------------------------===//
  // Manipulators
  //===--------------------------------------------------------------------===//

  /// Add a positional argument to the list.
  void add(ASTExprAnd<AnyValue> value) {
    values.emplace_back(StringAttr(), std::move(value));
  }

  /// Add a keyword argument, there can never be conflicts here because keyword
  /// argument conflicts should be checked in the parser before any semantic
  /// analysis is attempted.
  void add(StringAttr name, ASTExprAnd<AnyValue> value) {
    values.push_back({name, std::move(value)});
  }

  /// This adds a "self" argument to the start of the positional argument list.
  void addSelf(ASTExprAnd<AnyValue> value) {
    assert(!hasSelfOperand && "Cannot add a self when one is already present");
    values.insert(values.begin(), {StringAttr(), value});
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