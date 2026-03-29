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

namespace M::KGEN::LIT {
class PogListAttr;

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

/// When emitting a function call, this enum is used to indicate why the call
/// happened in the first place.  This allows producing better-tuned
/// diagnostics.
enum class CallSyntax : uint8_t {
  kParamBindings,      //< symbol[x, val=y]  (not actually a call).
  kDirectCall,         //< f()
  kIndirectCall,       //< expr()
  kMethodCall,         //< x.f()
  kTypeCall,           //< T()
  kOperator,           //< -x and x + y
  kReversedOperator,   //< y + x          (where the method was looked up on x).
  kSubscript,          // v[1, 2]
  kAttribute,          // v.x             (where x is not a static member of v).
  kImplicitConvert,    //< Conversion in an argument context
  kImplicitCopyCtor,   //< Implicit copy constructor call.
  kImplicitMoveCtor,   //< Implicit move constructor call.
  kDestructor,         //< Destructor due to a value definition.
  kTupleGetItem,       //< Call to getitem in a tuple assignment.
  kMethodCallSynthetic //< Call to a method for synthetic checks.
};

StringRef stringifyCallSyntax(CallSyntax val);
raw_ostream &operator<<(raw_ostream &os, CallSyntax val);

//===----------------------------------------------------------------------===//
// CallOperands
//===----------------------------------------------------------------------===//

/// This is an operand record, maintaining the IR repre that might
struct OperandValue : public ASTExprAnd<AnyValue> {
  // Null for positional arguments.
  StringAttr keyword;
  // True if the operand is passed via positional "unpack" (`*x`). This is not
  // used for keyword unpacking (`**x`).
  bool unpackedPositional = false;

  OperandValue(StringAttr keyword, ASTExprAnd<AnyValue> value,
               bool unpackedPositional = false)
      : ASTExprAnd<AnyValue>(std::move(value)), keyword(keyword),
        unpackedPositional(unpackedPositional) {
    assert((!unpackedPositional || !keyword) &&
           "unpacked positional operands cannot be keywords");
  }

  bool isUnpackedPositional() const { return unpackedPositional; }
};

using OperandValueList = SmallVector<OperandValue, 4>;

/// Struct that carries both positional and keyword operands for a call or
/// parameter binding. This does not own any values, only references pointers
/// to their containers.
class CallOperands {
public:
  /// Initialize with some positional arguments.
  CallOperands(CallSyntax syntax, const ExprNode *callExpr,
               ArrayRef<ASTExprAnd<AnyValue>> posOperands)
      : syntax(syntax), callExpr(callExpr) {
    for (const auto &operand : posOperands)
      values.emplace_back(StringAttr(), operand);
  }

  CallOperands(CallSyntax syntax, const ExprNode *callExpr)
      : syntax(syntax), callExpr(callExpr) {}
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

  /// This is the syntax the operand list is being used for.
  CallSyntax syntax;

  /// This is the expression representing the overall call.
  const ExprNode *callExpr;

  const ExprNode *getExpr() const { return callExpr; }
  llvm::SMLoc getExprLoc() const;

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

  /// Add a positional argument that came from `*expr`.
  void addUnpackedPositional(ASTExprAnd<AnyValue> value) {
    values.emplace_back(StringAttr(), std::move(value),
                        /*unpackedPositional=*/true);
  }

  /// Add a keyword argument, there can never be conflicts here because keyword
  /// argument conflicts should be checked in the parser before any semantic
  /// analysis is attempted.
  void addKeyword(StringAttr name, ASTExprAnd<AnyValue> value) {
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
    kOutOfOrderInferredKw,
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

//===----------------------------------------------------------------------===//
// OperandsNeedingOriginsList
//===----------------------------------------------------------------------===//

/// Parameter inference is used to evaluate whether a set of operands can work
/// for a callee, and determine a set of parameter bindings to use for it.
///
/// In that process, it may find that it could select the candidate if a
/// non-memory operand(eg a PValue or SRValue) were to be dumped into memory.
/// This list keeps track of those cases.
struct OperandNeedingOrigin {
  size_t operandIdx; // The index of the operand in the call operands list.
  size_t argIdx;     // The index of the argument in the callee's signature.
  ASTType expectedArgType; // The expected RValue type of the argument.
};
using OperandsNeedingOriginsList = std::vector<OperandNeedingOrigin>;

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_OPERANDCONTAINER_H
