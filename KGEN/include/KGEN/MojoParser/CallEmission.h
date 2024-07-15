//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CALLEMISSION_H
#define KGEN_MOJOPARSER_CALLEMISSION_H

#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/ParamBindings.h"
#include "KGEN/MojoParser/TypeCheckScopeInfo.h"

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// OperandContainer
//===----------------------------------------------------------------------===//

/// A shorthand to make positional operand handling more readable.
using FuncOperand = ASTExprAnd<AnyValue>;

/// A shorthand to make keyword operand handling more readable.
template <typename OperandType>
using KeywordOperandContainer =
    llvm::MapVector<StringAttr, OperandType, SmallDenseMap<StringAttr, size_t>>;

/// A shorthand to make keyword argument handling more readable.
using KeywordOperands = KeywordOperandContainer<FuncOperand>;

/// Struct that carries both positional and keyword operands for a call or
/// parameter binding. This does not own any values, only references pointers
/// to their containers.
template <typename OperandType>
class OperandContainer {
public:
  /// Create call operands with positional and optional keyword arguments.
  OperandContainer(
      ArrayRef<OperandType> posOperands = {},
      const KeywordOperandContainer<OperandType> *kwOperands = nullptr)
      : posOperands(posOperands), kwOperands(kwOperands) {}

  /// Create call operands with positional arguments given a value implicitly
  /// convertible to `ArrayRef`.
  template <typename OperandsT,
            typename = std::enable_if_t<
                !std::is_same_v<OperandsT, ArrayRef<OperandType>> &&
                std::is_convertible_v<OperandsT, ArrayRef<OperandType>>>>
  OperandContainer(
      OperandsT &&posOperands,
      const KeywordOperandContainer<OperandType> *kwOperands = nullptr)
      : OperandContainer(
            ArrayRef<OperandType>(std::forward<OperandsT>(posOperands)),
            kwOperands) {}

  /// Form a reference from parameter bindings.
  OperandContainer(const ParamBindings &bindings)
      : posOperands(bindings.posBindings), kwOperands(&bindings.kwBindings) {}

  /// Return a keyword argument value if present, or null otherwise.
  std::optional<OperandType> findKwArg(StringAttr argName) const {
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
  ArrayRef<OperandType> posOperands;

  /// The values passed as keyword operands.
  const KeywordOperandContainer<OperandType> *kwOperands;
};

//===----------------------------------------------------------------------===//
// CallOperands
//===----------------------------------------------------------------------===//

/// Struct that carries both positional and keyword operands for a call. This
/// does not own any values, only references pointers to their containers.
class CallOperands : public OperandContainer<FuncOperand> {
public:
  using OperandContainer::OperandContainer;

  /// Inidicates if the positional operands include a self operand.
  bool hasSelfOperand = false;

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, const CallOperands &value);

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

/// When emitting a function call, this enum is used to indicate why the call
/// happened in the first place.  This allows producing better-tuned
/// diagnostics.
enum class CallSyntax : uint8_t {
  kDirectCall,         //< f()
  kIndirectCall,       //< expr()
  kMethodCall,         //< x.f()
  kTypeCall,           //< T()
  kOperator,           //< -x and x + y
  kReversedOperator,   //< y + x          (where the method was looked up on x).
  kSubscript,          // v[1, 2]
  kAttribute,          // v.x             (where x is not a static member of v).
  kImplicitConvert,    //< Conversion in an argument context
  kDestructor,         //< Destructor due to a value definition.
  kTupleGetItem,       //< Call to getitem in a tuple assignment.
  kMethodCallSynthetic //< Call to a method for synthetic checks.
};

StringRef stringifyCallSyntax(CallSyntax val);
raw_ostream &operator<<(raw_ostream &os, CallSyntax val);

//===----------------------------------------------------------------------===//
// OverloadSet
//===----------------------------------------------------------------------===//

/// This class represents an unresolved overload set with partially bound
/// callees, e.g. "foo" or "a.foo" where "foo" is an overloaded declaration or
/// an incompletely bound function (e.g. one with result parameters).  This is
/// resolved when emitted to an RValue or when binding more things into it as
/// part of the expression tree.
///
/// Note that it is possible to have an overload set with methods from multiple
/// different self types that are related to each other.  For example when Mojo
/// has classes, it will be common to have super-class methods that expect
/// 'self' to be converted to a different type in order to invoke it.  For
/// nonmaterializable types like IntLiteral, we can have methods on both Int and
/// IntLiteral, etc.  Filtering the overload set will pick the appropriate
/// method.
class OverloadSet {
public:
  /// In a method reference like `x.foo`, this is the base object being invoked,
  /// e.g. `x`.
  ASTExprAnd<AnyValue> baseValue;

  /// This is the basename of the declaration set, used in diagnostics.
  StringRef baseName;

  /// The function overload set that may be called directly.
  SmallVector<ASTDecl *, 1> fnDecls;

  /// Any bound parameters.
  ParamBindings paramBindings;

  /// This is information about where this overload set was formed.
  const ExprNode *expr;
  CallSyntax syntax;

  /// When doing resolution, we should only raise new errors if previous errors
  /// haven't already been raised about functions in the overload set.  The most
  /// common issue is when one of the included declarations is erroneous.
  /// Emitting further errors about overload resolution failure can then be
  /// spurious, since we can't properly consider the erroneous declarations
  /// which otherwise might match.  This flag guards against raising those extra
  /// errors.
  bool erroneous;

  /// Form an overload set with the specified function overloads and the given
  /// parameter bindings. The parameter bindings are taken ownership of.
  OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
              ParamBindings &&paramBindings, const ExprNode *expr,
              CallSyntax syntax, bool erroneous = false);

  /// Form an OverloadSet with a lookup of a named method on the specified type,
  /// but without the candidate set filtered with operands.   If successful,
  /// this provides a non-null OverloadSet.
  ///
  /// On failure, this returns a null OverloadSet and invokes errorHandler if
  /// the problem hasn't already been diagnosed and it is non-null. This does
  /// not emit an error on failure.
  static OverloadSet lookup(const TypeCheckScopeInfo &scopeInfo, ASTType type,
                            StringRef methodName, const ExprNode *callExpr,
                            CallSyntax syntax,
                            function_ref<void()> errorHandler = {});

  /// Lookup of a named method on the specified type, filtered to match a
  /// concrete operand set. If successful, this provides a non-null PValue for a
  /// single callee. If non-null, it invokes lookupFailureErrorHandler if the
  /// lookup of the named method fails.  If that succeeds, it will complain
  /// about overload resolution when 'shouldPrintOverloadErrors' is true.
  static PValue lookupAndResolve(const TypeCheckScopeInfo &scopeInfo,
                                 ASTType type, StringRef methodName,
                                 CallOperands &callOperands,
                                 const ExprNode *callExpr, CallSyntax syntax,
                                 function_ref<void()> lookupFailureErrorHandler,
                                 bool shouldPrintOverloadErrors);

  /// Same as the above but a convenience when never emitting an error.
  static PValue lookupAndResolve(const TypeCheckScopeInfo &scopeInfo,
                                 ASTType type, StringRef methodName,
                                 CallOperands &callOperands,
                                 const ExprNode *callExpr, CallSyntax syntax) {
    return lookupAndResolve(scopeInfo, type, methodName, callOperands, callExpr,
                            syntax, {}, false);
  }

  bool isNull() const { return fnDecls.empty(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  /// An overload set is erroneous primarily when constructed with erroneous
  /// decls.  If an overload set is erroneous, you can't necessarily trust
  /// lookup results when processing to find further errors.
  bool isErroneous() const { return erroneous; }

  const TypeCheckScopeInfo &getScopeInfo() const { return paramBindings; }
  SharedState &getShared() const { return paramBindings.shared; }

  /// Perform substitutions of the specified bindings into the symbol, returning
  /// the resultant LITSymbolConstant attr or producing an error message and
  /// returning null. This allows producing a reference to a parameterized
  /// function without the parameters specified.  They can be bound later.
  TypedAttr getBoundConstantAttr() const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings and provided
  /// arguments.  If so, return the single entry that works and potentially
  /// mutate the operand list (when calling a static method that doesn't need
  /// a self value)).
  ///
  /// If not, generate a diagnostic (when `emitDiagnosticOnFailure` is true) and
  /// return null.
  PValue filterOverloadSet(CallOperands &operands,
                           bool allowImplicitConversions,
                           bool emitDiagnosticOnFailure) const;

  /// Evaluate the fnDecls candidates and see if there is an unambiguous
  /// candidate that works with the specified parameter bindings on the overload
  /// set. If so, return the single entry that works.  If not, generate a
  /// diagnostic and return null.
  PValue filterOverloadSetForParamBindings(bool allowImplicitConversions) const;

  /// Try to resolve the overload set to a single function candidate, using the
  /// expected type if provided or using current bindings if an emitter is
  /// provided.  This emits errors if 'emitter' is non-null, but does not if it
  /// is null.
  PValue getDirectSymbol(ASTType expectedType) const;

  /// Try to emit the overload set as a PValue.
  PValue getIfPValue() const;

  /// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
  /// error and return null.
  CValue emitAsCValue(ExprEmitter &emitter, ValueDest &dest);

  /// Emit a function call to the specified callee with the specified operand
  /// values.  This emits an error and returns null on failure.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitCall(const CallOperands &callOperands, ValueDest &dest,
                  ExprEmitter &emitter);

  /// Filter down and complete this overload set based on knowledge that we need
  /// to produce a function pointer with the specified type.  This returns a
  /// PValue for the callee if resolvable or null if not.
  PValue filterOverloadSetForValueType(ASTType functionType,
                                       bool emitDiagnosticOnFailure) const;
  PValue filterOverloadSetForValueType(
      ASTType functionType,
      function_ref<InflightDiag &(llvm::SMLoc)> emitError) const;

  /// If the specified type can be constructed with the specified operands
  /// return the initializer that would be invoked. If not, return null PValue.
  /// If there were erroneous declarations when processing return failure so we
  /// don't indicate downstream errors.
  ///
  /// If there were erroneous declarations, an error has been raised about a
  /// constructor that likely would have applied, which should be considered in
  /// any error reporting. This does not generate any IR.
  static FailureOr<PValue>
  canConstructType(ASTType requiredType, const CallOperands &operands,
                   const ExprNode *expr, const TypeCheckScopeInfo &scopeInfo,
                   bool allowImplicitConversions = true);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  static bool canImplicitlyConvertToType(ASTExprAnd<CValue> value,
                                         ASTType requiredType,
                                         const TypeCheckScopeInfo &scopeInfo);

  LLVM_DUMP_METHOD void dump() const;

private:
  OverloadSet(const TypeCheckScopeInfo &scopeInfo, const ExprNode *expr,
              CallSyntax syntax, bool erroneous)
      : paramBindings(scopeInfo), expr(expr), syntax(syntax),
        erroneous(erroneous) {}
};

/// This provides a wrapper around OverloadSet which is reference counted,
/// allowing OverloadSetUValue to maintain it while still being copyable.
struct OverloadSetUValue::OverloadSetWrapper
    : public NonAtomicallyReferenceCounted<OverloadSetWrapper> {

  OverloadSetWrapper(OverloadSet &&overloadSet)
      : overloadSet(std::move(overloadSet)) {}
  OverloadSet overloadSet;
};

//===----------------------------------------------------------------------===//
// OverloadSetUValue implementation details
//===----------------------------------------------------------------------===//

template <typename... Args>
inline OverloadSetUValue OverloadSetUValue::create(Args &&...args) {
  return OverloadSetUValue(takeRCRef(
      new OverloadSetWrapper(OverloadSet(std::forward<Args>(args)...))));
}

inline const OverloadSet &OverloadSetUValue::operator*() const {
  return storage.getPointer()->overloadSet;
}

inline OverloadSet &OverloadSetUValue::operator*() {
  return storage.getPointer()->overloadSet;
}

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CALLEMISSION_H
