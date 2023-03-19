//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPREMITTER_H
#define LIT_EXPREMITTER_H

#include "IRValues.h"
#include "LitExprNode.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
template <typename ValueType>
struct ASTExprAnd;
enum class SpecialFunctionKind : uint8_t;
enum class CallSyntax : uint8_t;
class ExprEmitter;
class VarLetDeclOp;

/// This enum is used to pass down a bit of context information to make
/// diagnostics more specific.  Each comment gives an example where the
/// expression is named "x".
enum ExprContext {
  EC_Unknown, // No context known.
  EC_Silent,  // Do not emit a diagnostic at all.

  EC_VarInit,               // var thing = x
  EC_LetInit,               // let thing = x
  EC_Assignment,            // y = x
  EC_Type,                  // var v : x         (and many other places)
  EC_AttributeRefBase,      // x.field
  EC_AliasValue,            // alias something = x
  EC_CallArgValue,          // foo(x)
  EC_CallCalleeValue,       // x()
  EC_TypeParamValue,        // Vector[x]
  EC_CallParamValue,        // f[x]()
  EC_OperatorOperandValue,  // x + y
  EC_InplaceBinOpDest,      // x += 42
  EC_FieldInitValue,        // SomeType{value: x}
  EC_DefaultArgument,       // def f(arg = x):
  EC_DefArgumentShadow,     // def f(x: Int):    -> var shadow slow.
  EC_BoolCondition,         // if x  /  while x  /  x and y  /  a if x else b
  EC_ForIterator,           // for x internal details
  EC_RaiseValue,            // raise x
  EC_ReturnResultParamList, // return[x] y
  EC_ReturnValue,           // return x;
  EC_MLIRMagic,             // __mlir_type[x] / __mlir_attr[x]
};
const char *getContextMessage(ExprContext context);

/// This is used in ValueDest when emitting an LValue expression whose type may
/// be inferred from the RHS value in an assignment.  This allows implicitly
/// declared variables and discard patterns to infer their type in `_ = foo()`.
struct LValueInitializerType {
  ASTType type;
};

/// This class represents the destination context than an expression is being
/// emitted, when it may produce an RValue.  Example destinations include:
///   - an LValue:
///       This handles cases like `a.b = 42` or `var x: Int = 42`, as well as
///       a return slot with memory-only results in `return x()`.  In this
///       case, the emitted expression must conform to type of the LValue.
///   - an untyped let/var decl, e.g. `var x = 42`
///       In this case, the ExprNode conforms to the initializer expression.
///   - an ExprNode:
///       This handles assignments to "targets" (Python nomenclature), e.g.:
///          1) a discard pattern, e.g. `_ = 42`
///          2) an implicitly declared var decl, e.g. `x = 42` in a def.
///          3) tuples and lists thereof, e.g. `(a, _) = foo()`
///       In this case, the ExprNode type often conforms to the expression.
///   - an RValue type:
///       This indicates that the result may be treated in any way (e.g. dumping
///       into a temporary memory location as an MRValue or returned in an SSA
///       register as an SRValue) but needs to have the specified RValue type.
///   - LValueInitializerType:
///       This is used when emitting LValues when there is an inferred type for
///       the LValue, e.g. in `_ = foo()`.
///
/// Any expression may also have no proscribed result (as in the case of
/// `someExpr()` with an ignored result), in which case emission will create
/// storage when needed on demand.
class ValueDest {
public:
  /*implicit*/
  ValueDest(ExprContext context = EC_Unknown)
      : representation(NullRepresentation()), context(context) {}
  ValueDest(const ExprNode *target, ExprContext context)
      : representation(target), context(context) {
    assert(target);
  }
  ValueDest(LValue dest, ExprContext context)
      : representation(dest), context(context) {}
  ValueDest(VarLetDeclOp dest, ExprContext context);
  ValueDest(ASTType requiredType, ExprContext context)
      : representation(requiredType), context(context) {
    if (!requiredType)
      representation = NullRepresentation();
  }
  ValueDest(LValueInitializerType type, ExprContext context)
      : representation(type), context(context) {}

  ValueDest(ValueDest &&rhs)
      : representation(std::move(rhs.representation)), context(rhs.context) {
    rhs.resetForError();
  }
  ValueDest &operator=(ValueDest &&rhs) {
    representation = std::move(rhs.representation);
    context = rhs.context;
    rhs.resetForError();
    return *this;
  }

  ValueDest(const ValueDest &) = delete;
  ValueDest &operator=(const ValueDest &) = delete;

  // This is a "constructor" that allows accessing an empty ValueDest by
  // reference.  By convention we know this will never get mutated, so the
  // reference is safe to share.
  static ValueDest &none() {
    static ValueDest dummy;
    return dummy;
  }

  ~ValueDest() {
    assert(!isSpecified() && "ValueDest destroyed without being emitted into");
  }

  /// This returns the context the expression is getting emitted into (for
  /// diagnostic QoI purposes).
  ExprContext getContext() const { return context; }

  /// Return true if there is a specification for this destination.  If not,
  /// an expression will be emitted to generate a PValue, SRValue, LValue, etc.
  bool isSpecified() const { return !isa<NullRepresentation>(representation); }

  /// Return the LValueInitializerType this contains if it is one.
  ASTType getIfLValueInitializerType() const {
    if (isa<LValueInitializerType>(representation))
      return cast<LValueInitializerType>(representation).type;
    return {};
  }

  /// Inspect the ValueDest to see if it implies a specific type for the value
  /// being computed, emiting ExprNode targets if present to get their implied
  /// type if present.  This returns null if there is no implied type.
  ///
  /// This may be used in concrete value context with a known type (in which
  /// case 'existingValueType' will hold the known value type) or in ambiguous
  /// cases where this is being used to resolve a type (in which case it will be
  /// null).
  ///
  /// Note that this will mutate the ValueDest if it is an ExprNode, turning it
  /// into an LValue to store to.
  ASTType resolveImpliedType(SMLoc loc, Type existingValueType,
                             ExprEmitter &emitter);

  /// Project a ValueDest into an lvalue with the specified underlying (RValue)
  /// type.
  ///
  /// When `allowIncompatibleTypes` is true, the method is allowed to return an
  /// LValue of a different type when the underlying storage requires this. This
  /// is a guarantee from the caller that it is prepared to handle a type
  /// conversion on its side, eliminating a temporary buffer in
  /// register-passable cases like `var x : F32 = 1`.
  ///
  /// When `allowIncompatibleTypes` is false, this always returns an LValue of
  /// the requested type, which may return a temporary buffer.  In this case it
  /// will not consume the ValueDest, so any user should reemit the ultimate
  /// value through it with emitResult.
  LValue getLValueForResult(SMLoc loc, ASTType resultType,
                            bool allowIncompatibleTypes, bool requireSLValue,
                            ExprEmitter &emitter);

  /// Return an SLValue for this destination of the specified type that we can
  /// initialize.  This uses and consumes the destination if it matches the type
  /// of the value dest.
  SLValue getSLValueForResult(SMLoc loc, ASTType resultType,
                              ExprEmitter &emitter);

  /// When an error is emitted instead of generating IR, this method resets the
  /// ValueDest so it doesn't complain when emission is done.
  void resetForError() { representation = NullRepresentation(); }

private:
  //  This should only be accessed by ExprEmitter::emitResult.
  friend class ExprEmitter;
  SmartVariant<NullRepresentation, LValue, const ExprNode *, Operation *,
               ASTType, LValueInitializerType>
      representation;
  ExprContext context;
};

/// This class is the main driver for expression emission, providing helper
/// functions used by the individual node emission hooks.
class ExprEmitter : public LitSharedStateUser {
public:
  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              std::optional<OpBuilder> builder, Operation *varDeclCursor)
      : LitSharedStateUser(shared), builder(builder), declScope(declScope),
        varDeclCursor(varDeclCursor) {}

  //===--------------------------------------------------------------------===//
  // Emitter State.

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  std::optional<OpBuilder> builder;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// When non-null, implicitly declared variables are added above this op.
  Operation *varDeclCursor;

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This helper emits the specified value as an RValue.
  RValue emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  RValue emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType);
  CRValue emitCRValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  CValue emitCValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  LValue emitLValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  BValue emitBValue(ASTExprAnd<AnyValue> value, ValueDest &dest);

  /// This helper emits the specified value as a SRValue which has an SSA
  /// value representation, materializing PValues and loading LValues as
  /// needed.  This returns null if emission fails, and should never be used
  /// with values that are memory-only.
  SRValue emitSRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                      ASTType resultType = {});

  /// This helper emits the specified value as a PValue. This returns null if
  /// emission fails.
  PValue emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});

  //===--------------------------------------------------------------------===//
  // Function Calls

  /// This helper emits a named method call with the provided `argValues`, where
  /// the first arg is the receiver of the call. This emits an error if the
  /// call is invalid and returns null.  The argValues list may not be empty.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitNamedMethodCall(StringRef methodName,
                             ArrayRef<ASTExprAnd<AnyValue>> argValues,
                             ValueDest &dest, CallSyntax syntax,
                             const ExprNode *callNode);

  /// Emit an indirect call to a resolved value, checking for compatibility and
  /// then generating the call logic.  This emits an error and returns null on
  /// failure.
  CValue emitIndirectCall(CRValue callee,
                          ArrayRef<ASTExprAnd<AnyValue>> operands,
                          ValueDest &dest, const ExprNode *callExpr);

  /// Emit call to a resolved and /already type checked/ callee. This does not,
  /// check for compatibility and isn't prepared to emit errors.
  CValue emitCallUnchecked(CRValue callee,
                           ArrayRef<ASTExprAnd<AnyValue>> operands,
                           ArrayRef<ParamDeclAttr> resultParams,
                           ValueDest &dest, const ExprNode *callExpr);

  /// Return true if 'value' may be implicitly converted to 'requiredType'
  /// by invoking (one level of) conversion operations.  This does not generate
  /// any IR.
  bool canImplicitlyConvertToType(ASTExprAnd<CValue> value,
                                  ASTType requiredType);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly, and also returning the intermediate
  /// result of calling `__bool__` (which is typically a Bool or object type,
  /// but not guaranteed).  This reports and error and returns null on error.
  RValue emitConditionValueAsI1(ASTExprAnd<CValue> expr, CValue &boolResult);

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// Emit the specified value into the current destination if present.  This
  /// accepts (and silently propagates) null values.
  AnyValue emitResult(AnyValue value, const ExprNode *node, ValueDest &dest);
  CValue emitCResult(CValue value, const ExprNode *node, ValueDest &dest);

  /// This emits the specified value to an RValue in the specified ValueDest and
  /// returns it (potentially as a borrowed referenced to that storage).
  RValue emitExprRValue(const ExprNode *node, ValueDest &dest);

  /// This emits the specified value rep as an RValue.
  CRValue emitExprCRValue(const ExprNode *node, ValueDest &dest);

  /// This helper emits the specified value rep as an SRValue, materializing
  /// it as an operation if it is a parameter.  This returns null if emission
  /// fails.
  SRValue emitExprSRValue(const ExprNode *node, ExprContext context,
                          ASTType resultType = {});

  /// This helper emits the specified expression as a meta value, and optionally
  /// converts the result to a specified expected type.  This emits an error if
  /// the expression cannot be emitted, if it cannot be converted to the
  /// expected type, or if it isn't a valid runtime value.  This returns null if
  /// emission fails.
  PValue emitExprPValue(const ExprNode *node, ExprContext context,
                        ASTType resultType = {});

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// The ValueDest may specify an inferred type for the LValue.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitExprLValue(const ExprNode *expr, ValueDest &dest);
  LValue emitExprLValue(const ExprNode *expr, ExprContext context) {
    ValueDest dest(context);
    return emitExprLValue(expr, dest);
  }

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This emits an error and returns null on
  /// failure.  If `isPack` is true, then values of variadic type are lowered
  /// into a pack type.
  ASTType emitExprType(const ExprNode *expr, bool isPack);

  /// Emit a call to __new__ or __init__, returning an instance of the specified
  /// type.  If `allowImplicitConversion` is true, the provided args are allowed
  /// to implicitly convert to the expectations of the constructor signatures.
  CValue emitConstructorCall(ASTType type, ArrayRef<ASTExprAnd<AnyValue>> args,
                             const ExprNode *expr, CallSyntax syntax,
                             ValueDest &dest,
                             std::function<void()> errorHandler = {},
                             bool allowImplicitConversion = true);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly.  This reports and error and returns null
  /// on error.
  RValue emitExprConditionValueAsI1(const ExprNode *condExpr);

  /// Given a value convertable to a pop int via index conversion, emit
  /// the casting code and return the pop scalar index value
  SRValue emitBoxedIntAsPopScalar(Value numberValue, const ExprNode *source);

  /// Given an BValue, produce a standalone rvalue in the specified destination
  /// by emitting a clone call.
  RValue emitBValueToRValue(ASTExprAnd<BValue> value, ValueDest &dest);
};

} // namespace M::KGEN::LIT

#endif // LIT_EXPREMITTER_H
