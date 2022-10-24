//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides machinery used when emitting expressions to MLIR, either
// as operations for runtime values or as attributes for metavalues.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPRS_H
#define LIT_EXPRS_H

#include "LitSharedState.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ExprNode;
class Scope;

/// When emitting an expression to MLIR as an rvalue, we get a value back that
/// is either an attribute (for parameter expressions) or an SSA value.  The
/// stored attribute is always actually a TypedAttr.
class MLIRValueRep : public PointerUnion<Attribute, Value> {
public:
  using Base = PointerUnion<Attribute, Value>;
  using Base::PointerUnion;

  /// If this contains an Attribute, it is known to be a TypedAttr.  This helper
  /// performs the conversion.  This returns null if this contains a value.
  TypedAttr dyn_castTypedAttr() const;

  /// Return the type for the contained TypedAttr or Value, or null if they are
  /// both null.
  Type getType() const;
};

//===----------------------------------------------------------------------===//
// ExprEmitter
//===----------------------------------------------------------------------===//

struct ExprEmitter {
  /// This is the shared state for the parser overall.
  LitSharedState &shared;

  /// This is scope to resolve declaration references against.
  Scope &scope;

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  Optional<OpBuilder> builder;

  ExprEmitter(LitSharedState &shared, Scope &scope, Optional<OpBuilder> builder)
      : shared(shared), scope(scope), builder(builder) {}

  MLIRContext *getContext() const { return shared.context; }

  /// This helper emits the specified value rep as an SSA value, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  Value emitAsValue(MLIRValueRep rep, SMLoc loc);

  /// This helper emits the specified value rep as an SSA value, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  Value emitAsValue(const ExprNode *node);

  /// This helper emits the specified value rep as a meta value, diagnosing the
  /// problem if the expression is only valid as a runtime value (using the
  /// specified message).  This returns null if emission fails.
  TypedAttr emitAsMetaValue(const ExprNode *node, const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null - if the expression
  /// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
  Type emitAsType(const ExprNode *node);

  /// Perform a name lookup in the current scope and return the named
  /// declaration.  This emits an error and returns null on error.
  Scope *lookupDecl(StringRef name, SMLoc loc);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &twine) const {
    shared.errorOccurred = true;
    return mlir::emitError(translateLocation(loc), twine);
  }

  /// Translate an SMLoc into an MLIR Location.
  Location translateLocation(SMLoc loc) const {
    return shared.translateLocation(loc);
  }
};

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
class ExprNode {
public:
  // This indicates the subclass.
  enum Kind {
    kError,         // `
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kStringLiteral, // "Hello"
    kDeclRef,       // x
    kCall,          // thing(a, b)
    kSubscript,     // thing[a, b:c]
    kParenExprNode, // (x+y)

    // Binary expressions.
    kAdd,
    kMul,
    kFirstBinOp = kAdd,
    kLastBinOp = kMul,

  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;

  /// Return true if this expression tree contains an already-reported error.
  virtual bool containsError() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) MLIRValueRep.
  virtual MLIRValueRep emitIR(ExprEmitter &state) const = 0;

  /// Emit this expression tree to an MLIR type.  This returns null on error,
  /// unlike the corresponding ExprEmitter method.
  virtual Type emitType(ExprEmitter &state) const = 0;
};

} // namespace M::KGEN::LIT

namespace llvm {

template <typename To>
struct CastInfo<To, const M::KGEN::LIT::MLIRValueRep>
    : public CastInfo<To, const M::KGEN::LIT::MLIRValueRep::Base> {};
template <typename To>
struct CastInfo<To, M::KGEN::LIT::MLIRValueRep>
    : public CastInfo<To, M::KGEN::LIT::MLIRValueRep::Base> {};

} // namespace llvm

#endif // LIT_EXPRS_H
