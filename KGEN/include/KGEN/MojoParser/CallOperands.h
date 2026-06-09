//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CALLOPERANDS_H
#define KGEN_MOJOPARSER_CALLOPERANDS_H

#include "KGEN/MojoParser/ExprDest.h"

namespace M::KGEN {
class PogListAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {

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
  // This indicates whether the operand is a keyword argument or a positional
  // argument and if it is unpacked.
  ArgUnpackStyle unpackStyle;

  OperandValue(StringAttr keyword, ASTExprAnd<AnyValue> value,
               ArgUnpackStyle unpackStyle)
      : ASTExprAnd<AnyValue>(std::move(value)), keyword(keyword),
        unpackStyle(unpackStyle) {
    assert((keyword != StringAttr()) ==
               (unpackStyle == ArgUnpackStyle::kKeyword) &&
           "Keyword is present iff keyword argument");
  }
};

using OperandValueList = SmallVector<OperandValue, 4>;

/// Struct that carries information necessary to look up and emit functions and
/// method calls. This includes the destination to emit into, the operands (both
/// positional and keyword), and the syntax of the call.
class CallOperands {
public:
  /// Initialize with the call. The syntax and an expression node are required
  /// this constructor supports an optional list of positional operands as a
  /// convenience, but those can be added later as well.
  CallOperands(CallSyntax syntax, const ExprNode *callExpr, ExprDest &&dest,
               ArrayRef<ASTExprAnd<AnyValue>> posOperands = {})
      : syntax(syntax), callExpr(callExpr), dest(std::move(dest)) {
    for (const auto &operand : posOperands)
      add(operand);
  }

  // Initialize with an existing CallOperands and a new destination.
  CallOperands(const CallOperands &existing, ExprDest &&dest)
      : syntax(existing.syntax), callExpr(existing.callExpr),
        dest(std::move(dest)), values(existing.values),
        hasSelfOperand(existing.hasSelfOperand) {}

  CallOperands(CallOperands &&) = default;
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

  /// This is the location the call is going to be emitted into.  This can
  /// include information about the expected result type, the origin of the
  /// destination etc.
  ExprDest dest;

  /// The values passed in.  The keyword field will be null for positional
  /// arguments and present for keyword operands.
  OperandValueList values;

  /// Indicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

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
  void add(ASTExprAnd<AnyValue> value,
           ArgUnpackStyle unpackStyle = ArgUnpackStyle::kPositional) {
    values.emplace_back(StringAttr(), std::move(value), unpackStyle);
  }

  /// Add a keyword argument, there can never be conflicts here because keyword
  /// argument conflicts should be checked in the parser before any semantic
  /// analysis is attempted.
  void add(StringAttr name, ASTExprAnd<AnyValue> value,
           ArgUnpackStyle unpackStyle) {
    values.push_back({name, std::move(value), unpackStyle});
  }

  /// This adds a "self" argument to the start of the positional argument list.
  void addSelf(ASTExprAnd<AnyValue> value) {
    assert(!hasSelfOperand && "Cannot add a self when one is already present");
    values.insert(values.begin(),
                  {StringAttr(), value, ArgUnpackStyle::kPositional});
    hasSelfOperand = true;
  }

  //===--------------------------------------------------------------------===//
  // Diagnostic helpers.
  //===--------------------------------------------------------------------===//

  void dump() const;

  /// Helper to diagnose common cases of candidate mismatch related to keyword
  /// operands (unexpected kw-operands, pos-only arg/param provided by
  /// kw-operand, missing kw-only arg/param). This function collects any
  /// variadic keyword args/params if the function allows them.
  LogicalResult diagnoseKeywordOperands(
      PogListAttr pogListAttr, OperandValueList &variadicKwOperands,
      bool isParameterList,
      llvm::function_ref<MojoInflightDiag &()> getDiag) const;

  /// Helper to diagnose common cases of candidate mismatch related to
  /// positional arguments/parameter (too many positionals, missing positionals,
  /// argument/parameter specified both by positional and keyword operands).
  LogicalResult
  diagnosePosOperands(PogListAttr pogListAttr, bool isParameterList,
                      llvm::function_ref<MojoInflightDiag &()> getDiag) const;
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

  enum {
    /// This is a sentinel representing the the "operand" that needs spilling is
    /// actually the ExprDest of the call, not an actual operand.
    kExprDestOperandIdx = ~1ULL,
  };
};

using OperandsNeedingOriginsList = std::vector<OperandNeedingOrigin>;

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CALLOPERANDS_H
