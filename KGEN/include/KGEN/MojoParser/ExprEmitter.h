//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPREMITTER_H
#define KGEN_MOJOPARSER_EXPREMITTER_H

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoParser/TypeCheckScopeInfo.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
template <typename ValueType>
struct ASTExprAnd;
enum class SpecialFunctionKind : uint8_t;
enum class CallSyntax : uint8_t;
class ExprEmitter;
class CallOperands;
class AliasDeclOp;
class TraitType;
class VarDeclOp;

//===----------------------------------------------------------------------===//
// ExprContext
//===----------------------------------------------------------------------===//

/// This enum is used to pass down a bit of context information to make
/// diagnostics more specific.  Each comment gives an example where the
/// expression is named "x".
enum ExprContext {
  EC_InvalidContext,        // Not a valid context, will abort.
  EC_VarInit,               // var thing = x
  EC_Assignment,            // y = x
  EC_Type,                  // var v : x         (and many other places)
  EC_AttributeRefBase,      // x.field
  EC_AliasValue,            // alias something = x
  EC_CallArgValue,          // foo(x)
  EC_CallRefArgValue,       // foo(x) where x is passed by 'ref'.
  EC_CallCalleeValue,       // x()
  EC_TypeParamValue,        // Vector[x]
  EC_CallParamValue,        // f[x]()
  EC_OperatorOperandValue,  // x + y
  EC_InplaceBinOpDest,      // x += 42
  EC_FieldInitValue,        // SomeType{value: x}
  EC_DefaultArgument,       // def f(arg = x):
  EC_OwnedRegArgShadow,     // def f(x: Int):    -> var shadow slot.
  EC_VarArgArgument,        // fn f(x: *Int):    -> creation of VariadicList.
  EC_PackArgument,          // fn f[..](*x: *Ts) -> creation of VariadicPack
  EC_KWArgsArgument,        // fn f(x: **Int):   -> creation of KWArgs dict
  EC_DefaultParam,          // fn f[p: Int = x]():
  EC_BoolCondition,         // if x  /  while x  /  x and y  /  a if x else b
  EC_CondExpr,              // x if a else y
  EC_BoolParamCondition,    // @parameter if x
  EC_ForParamSeq,           // @parameter for y in x
  EC_ForIterator,           // for x internal details
  EC_WithContextMgr,        // with x:
  EC_WithExitResult,        // with (result of __exit__ call)
  EC_RaiseValue,            // raise x
  EC_ReturnResultParamList, // return[x] y
  EC_ReturnValue,           // return x;
  EC_MLIRMagic,             // __mlir_type[x] / __mlir_attr[x]
  EC_TopLevelStmt,          // x
  EC_ListField,             // [x, y]
  EC_TupleElement,          // (x, y)
  EC_SubscriptBase,         // x[y]
  EC_Subscript,             // y[x]
  EC_SliceIndex,            // y[:x:]
  EC_ParameterList,         // something[x]
  EC_Destructor,            // Looking up T's destructor for `var x : T`
  EC_Capture,               // def f(): var x = 4; def nested(): use(x)
  EC_Decorator,             // @x
  EC_AutoDeref,             // dereference Reference x
  EC_Trait,                 // trait conformance checking for `T`
  EC_Closure,               // closure formation
  EC_Lifetime,              // lifetime specifier
};
const char *getContextMessage(ExprContext context);

//===----------------------------------------------------------------------===//
// ValueDest
//===----------------------------------------------------------------------===//

/// This is used in ValueDest when emitting an LValue expression whose type may
/// be inferred from the RHS value in an assignment.  This allows implicitly
/// declared variables and discard patterns to infer their type in `_ = foo()`.
struct LValueInitializerType {
  ASTType type;
};

/// This is a marker type to indicate when a ValueDest has had its internal
/// LValue taken by getLValueForResult, but which hasn't had an emitResult to
/// write it back yet.
struct LValueBufferTaken {};

/// This class represents the destination context that an expression is being
/// emitted in, when it may produce an RValue.  Example destinations include:
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
  ValueDest(ExprContext context)
      : representation(NullRepresentation()), context(context) {}
  ValueDest(const ExprNode *target, ExprContext context)
      : representation(target), context(context) {
    assert(target);
  }
  ValueDest(LValue dest, ExprContext context)
      : representation(dest), context(context) {}
  ValueDest(VarDeclOp dest, ExprContext context);
  ValueDest(GlobalVarDeclOp dest, ExprContext context);
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

  /// If this indicates an explicit expected RValue type, return that type.
  ASTType getExpectedTypeIfSpecified() const;

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
                            bool allowIncompatibleTypes, bool requireMLValue,
                            ExprEmitter &emitter);

  /// Return an MLValue for this destination of the specified type that we can
  /// initialize.  This uses and consumes the destination if it matches the type
  /// of the value dest.
  MLValue getMLValueForResult(SMLoc loc, ASTType resultType,
                              ExprEmitter &emitter);

  /// If this ValueDest specifies an MLValue that will be returned by
  /// getMLValueForResult with the specified type, return it. Otherwise return
  /// null.
  MLValue getDefinedMLValueIfExists(ASTType resultType, ExprEmitter &emitter);

  /// Return true if this is an MLValue that could be in a non-default address
  /// space.
  bool isNonDefaultAddressSpace() const;

  /// When an error is emitted instead of generating IR, this method resets the
  /// ValueDest so it doesn't complain when emission is done.
  void resetForError() { representation = NullRepresentation(); }

  void dump() const;

private:
  //  This should only be accessed by ExprEmitter::emitResult.
  friend class ExprEmitter;
  SmartVariant<NullRepresentation, LValue, LValueBufferTaken, const ExprNode *,
               Operation *, ASTType, LValueInitializerType>
      representation;
  ExprContext context;
  friend raw_ostream &operator<<(raw_ostream &os, const ValueDest &value);
};

//===----------------------------------------------------------------------===//
// ExprEmitter
//===----------------------------------------------------------------------===//

/// This class is the main driver for expression emission, providing helper
/// functions used by the individual node emission hooks.
class ExprEmitter : public SharedStateUser {
public:
  /// Create an ExprEmitter for a dynamic context with a builder.
  ExprEmitter(SharedState &shared, ASTDecl &declScope, OpBuilder builder,
              std::optional<OpBuilder> varDeclCursor = {})
      : SharedStateUser(shared), builder(builder),
        paramContext(EC_InvalidContext), declScope(declScope),
        varDeclCursor(varDeclCursor) {}

  /// Create an ExprEmitter for a parameter context.
  ExprEmitter(SharedState &shared, ASTDecl &declScope, ExprContext paramContext)
      : SharedStateUser(shared), builder({}), paramContext(paramContext),
        declScope(declScope) {}

  /// Get an emitter set up for parameter expressions only with the specified
  /// context.
  ExprEmitter getParamEmitter(ExprContext context) {
    return ExprEmitter(shared, declScope, context);
  }

  //===--------------------------------------------------------------------===//
  // Emitter State.

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  std::optional<OpBuilder> builder;

  /// When builder is null, this specifies the original reason we're emitting
  /// into a parameter context, e.g. we're in an alias body or type
  /// specification.
  ExprContext paramContext;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// If specified, implicitly declared variables are added after this iterator.
  std::optional<OpBuilder> varDeclCursor;

  /// Return information about the scope we're looking into.
  TypeCheckScopeInfo getScopeInfo() const {
    return TypeCheckScopeInfo{declScope, shared};
  }

  /// Emit an error about use of a dynamic value (the expression) in a context
  /// that only allows parameter expressions.  This always returns a null
  /// PValue.
  PValue emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                             const char *customMessage = {});
  PValue emitErrorForDynamicValueInParameter(Location loc,
                                             const char *customMessage = {});

  /// If needed, convert the specified value to the target destination type,
  /// with a noop cast.  This is used to adjust inconsequential details of the
  /// type or for simple things like upcasts.  This does not invoke constructors
  /// or do other non-trivial conversions.
  ///
  /// This produces an error and returns null on an invalid conversion.
  AnyValue rebindValue(ASTExprAnd<AnyValue> value, Type destType);

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// This emits the value to the specified value dest, transferring ownership
  /// to the destination and returning a reference if dest consumes it, or the
  /// RValue directly if not.

  /// This emits the value to the specified destination as a concrete RValue.
  /// This transfers ownership to the destination, and it will return a
  /// reference if the destination consumes the RValue or the RValue itself if
  /// not. This method will also emit a copy if required to obtain and RValue.
  CValue emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  RValue emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  CValue emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  CValue emitCValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  BValue emitBValue(ASTExprAnd<AnyValue> value, ValueDest &dest);
  BValue emitBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});
  LValue emitLValue(ASTExprAnd<AnyValue> value, ValueDest &dest);

  /// Emit a register primary PValue to an SRValue.
  SRValue emitPValueToSRValue(ASTExprAnd<PValue> value, ExprContext context);
  /// Emit any kind of PValue to an MLValue.
  MBValue emitPValueToMLValue(ASTExprAnd<PValue> value, MLValue dest,
                              ExprContext context);
  /// This helper emits a PValue to an MRValue that has a memory representation,
  /// materializing the PValue.
  MRValue emitPValueToMRValue(ASTExprAnd<PValue> value, ExprContext context);

  /// This helper emits the specified value as a SRValue which has an SSA
  /// value representation, materializing PValues and loading LValues as
  /// needed.  This returns null if emission fails, and should never be used
  /// with values that are memory-only.
  SRValue emitSRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                      ASTType resultType = {});

  /// This helper emits the specified value as an MRValue which has
  /// memory-only representation, materializing PValues as needed. This
  /// returns null if emission fails.
  MRValue emitMRValue(ASTExprAnd<AnyValue> value, ExprContext context);

  /// This helper emits the specified value as an MBValue which has
  /// memory-only representation, materializing PValues as needed. This
  /// returns null if emission fails.
  MBValue emitMBValue(ASTExprAnd<AnyValue> value, ExprContext context);

  /// This helper emits the specified expression as a parameter value,
  /// diagnosing the problem if the expression is only valid as a runtime value.
  /// This returns null if emission fails.
  PValue emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
                    ASTType resultType = {});

  /// This helper emits the specified expression as a 'ref' expression value,
  /// and returns the value of RefType for the result.
  /// This emits an error and returns null if emission fails.
  Value emitRefValue(ASTExprAnd<AnyValue> value, ExprContext context);

  //===--------------------------------------------------------------------===//
  // Function Calls

  /// Emit an indirect call to a resolved value, checking for compatibility and
  /// then generating the call logic.  This emits an error and returns null on
  /// failure.
  CValue emitIndirectCall(CValue callee, CallOperands &&operands,
                          ValueDest &dest, const ExprNode *callExpr);

  /// This helper emits a named method call with the provided `operands`,
  /// where the first positional operand is the receiver of the call. This emits
  /// an error if the call is invalid and returns null. The `operands` must
  /// contain at least one positional operand.
  ///
  /// `callNode` is the call like expression (e.g. a CallNode, binary operator,
  /// etc) that results in the call, or potentially a random value that is being
  /// fed into an implicit conversion.  This should only be used for location
  /// information.
  CValue emitNamedMethodCall(StringRef methodName, CallOperands &&operands,
                             ValueDest &dest, CallSyntax syntax,
                             const ExprNode *callNode);

  /// Emit a call to __new__ or __init__, returning an instance of the specified
  /// type.  If `allowImplicitConversion` is true, the provided args are allowed
  /// to implicitly convert to the expectations of the constructor signatures.
  CValue emitConstructorCall(ASTType type, CallOperands &&operands,
                             const ExprNode *expr, CallSyntax syntax,
                             ValueDest &dest,
                             bool allowImplicitConversion = true);

  //===--------------------------------------------------------------------===//
  // Type conversion helpers.

  /// Emit a conversion from an MLIR type to a trait type by materializing stubs
  /// for the type's witness table.
  PValue bindMLIRTypeToTrait(ASTExprAnd<CValue> value, TraitType trait);

  /// Emit a metatype conversion to a trait type by materializing the type's
  /// witness table for the trait.
  PValue emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                       TraitType trait);

  /// This returns an instance of Tuple[...] with the specified element types
  /// installed.
  ASTType getBuiltinTupleInstantiation(llvm::SMLoc loc,
                                       ArrayRef<Type> elements);

  //===--------------------------------------------------------------------===//
  // Emission helpers for various value classifications.

  /// Emit the specified value into the current destination if present.  This
  /// accepts (and silently propagates) null values.
  ///
  /// Note that the `value` provided here may require an implicit conversion
  /// into the destination slot, so the input may be memory-only and result be
  /// register-passable (and visa-versa).
  AnyValue emitResult(AnyValue value, const ExprNode *expr, ValueDest &dest);
  CValue emitCResult(CValue value, const ExprNode *expr, ValueDest &dest);

  /// Emit the specified expression into the specified destination.
  AnyValue emitExpr(const ExprNode *expr, ValueDest &dest);

  /// Emit the specified node with the indicated expression context and an
  /// optional contextual type.
  AnyValue emitExpr(const ExprNode *expr, ExprContext context,
                    ASTType resultType = {});

  /// This emits the specified value to a RValue with the specified context.
  RValue emitExprRValue(const ExprNode *expr, ExprContext context,
                        ASTType resultType = {});

  /// This emits the specified value rep as a CValue.
  CValue emitExprCValue(const ExprNode *expr, ExprContext context,
                        ASTType resultType = {});

  /// This helper emits the specified value rep as an SRValue, materializing
  /// it as an operation if it is a parameter.  This returns null if emission
  /// fails.
  SRValue emitExprSRValue(const ExprNode *expr, ExprContext context,
                          ASTType resultType = {});

  /// This helper emits the specified expression as a meta value, and optionally
  /// converts the result to a specified expected type.  This emits an error if
  /// the expression cannot be emitted, if it cannot be converted to the
  /// expected type, or if it isn't a valid runtime value.  This returns null if
  /// emission fails.
  PValue emitExprPValue(const ExprNode *expr, ExprContext context,
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

  /// Emit a call to the getter of the specified LValue, loading the value into
  /// dest (if specified) or returning it if not.  This returns an RValue if
  /// there is no consuming dest, otherwise a BValue.
  CValue emitLoadOfLValue(ASTExprAnd<LValue> value, ValueDest &dest);

  /// Emit a copy of the specified value, producing a new owned instance of the
  /// value in the specified destination.  This returns an RValue if
  /// there is no consuming dest, otherwise a BValue.
  CValue emitCopyOfValue(ASTExprAnd<CValue> value, ValueDest &dest);

  /// Given a value with a known type, emit a store to the specified LValue.
  /// This returns an borrowed reference to the value after it is done.  The
  /// types must match for this call, or be a nonmaterializable conversion.
  BValue emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                           ExprContext context);

  //===--------------------------------------------------------------------===//
  // Emission helpers for specific value types.

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This emits an error and returns null on
  /// failure. If `allowUnbound` is set, then a type with no bound paramaters is
  /// allowed.
  ASTType emitExprType(const ExprNode *expr, bool allowUnbound = false);

  /// Emit the specified expression as a condition, converting it to an MLIR
  /// I1 value that we can test directly.  This reports and error and returns
  /// null on error.
  RValue emitI1(ASTExprAnd<CValue> value, ExprContext context);

  /// Emit the specified expression as a condition, converting it to an MLIR I1
  /// value that we can test directly.  This reports and error and returns null
  /// on error.
  RValue emitExprI1(const ExprNode *condExpr, ExprContext context);

  /// Given a value, emit it into an index value by invoking its `__index__`
  /// method.
  CValue emitIndex(ASTExprAnd<AnyValue> value, ExprContext context);

  /// Given a value, emit it into an MLIR value by convert it to an index value
  /// and then invoking its `__mlir_index__` method.
  CValue emitMLIRIndex(ASTExprAnd<AnyValue> value, ExprContext context);

  /// Emit the specified expression, converting it into an MLIR value by convert
  /// it to an index value and then invoking its `__mlir_index__` method.
  CValue emitMLIRIndex(const ExprNode *expr, ExprContext context);

  //===--------------------------------------------------------------------===//
  // Return emission helpers.

  /// Find the nearest error slot to use if the emitter is currently within a
  /// context that can raise. Otherwise, return null.
  MLValue findNearestErrorSlot();

  /// Emit a normal return (not a 'raise' return) out of the function, along
  /// with any special logic that goes with it.  `funcDecl` indicates the
  /// function we are returning out of.
  static void emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                               const ASTDecl &funcDecl);
  static void emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                               FuncOp funcDecl);

  //===--------------------------------------------------------------------===//
  // Var/let emission helpers.

  /// Helper to emit a VarDeclOp with a uniquely generated lifetime name.
  VarDeclOp emitVarDecl(const Twine &name, Type type, Location loc,
                        VarDeclKind kind);
  VarDeclOp emitVarDecl(StringAttr name, Type type, Location loc,
                        VarDeclKind kind);

  // Emit the vardecl shadow for an OwnedInReg argument.
  VarDeclOp makeArgLValueVarSlot(CValue argValue, StringAttr argName,
                                 SMLoc loc);

  /// Internal implementation of call emission, use emitCall/emitIndirectCall
  /// or higher level wrappers instead.
  CValue emitCallUnchecked(RValue callee, const CallOperands &operands,
                           ValueDest &dest, const ExprNode *callExpr);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPREMITTER_H
