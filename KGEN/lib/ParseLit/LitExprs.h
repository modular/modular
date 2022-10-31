//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides machinery used when emitting expressions to MLIR, either
// as operations for runtime values or as attributes for metavalues.
//
// Emitting an expression to MLIR can either produce a meta-value as an rvalue,
// may produce a runtime value as an rvalue, or may produce a metavalue as an
// LValue.  These make up the following hierarchy of value kinds:
//
//   AnyValue               <- Expr emitted to MLIR
//     LValue               <- Expr with a runtime address.
//     RValue               <- Expr without an address
//       MValue (TypedAttr) <- Expr with a meta-value (known at compile time)
//       DRValue (Value)    <- Expr with a dynamic value (only known at runtime)
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
class ExprEmitter;
class ASTDecl;

/// Instances of LValue model a dynamic address, which will always have pointer
/// type.  It is described with an explicit type because the type of the
/// underlying MLIR value may be pointer type for RValues of pointer type, so we
/// need something explicit to represent this.  This also helps avoid subtle
/// bugs in the emission phase.
struct LValue : public Value {
  using Value::Value;
  LValue(Value v) : Value(v) {}
};

/// This represents an RValue, which may be either a meta value (MValue)
class RValue {
public:
  RValue() : storage() {}
  RValue(Attribute metaValue) : storage(metaValue) {
    assert(isa<TypedAttr>(metaValue));
  }
  RValue(TypedAttr metaValue) : storage((Attribute)metaValue) {}
  RValue(Value rValue) : storage(rValue) {}
  explicit RValue(PointerUnion<Attribute, Value> storage) : storage(storage) {}

  bool isNull() const { return storage.isNull(); }
  bool operator!() const { return isNull(); }
  operator bool() const { return !isNull(); }

  /// If this contains a metavalue, return it; otherwise return null.
  TypedAttr getIfMValue() const {
    // Meta values are stored as Attribute because they are a single word, but
    // we know they always hold a TypedAttr.
    if (auto attr = dyn_cast<Attribute>(storage))
      return cast<TypedAttr>(attr);
    return {};
  }

  Value getIfDRValue() const { return dyn_cast<Value>(storage); }

  /// Return the type for the contained representation, or null if they are
  /// both null.
  Type getType() const;

  PointerUnion<Attribute, Value> getStorage() const { return storage; }

private:
  PointerUnion<Attribute, Value> storage;
};

class AnyValue {
public:
  AnyValue() : storage() {}
  AnyValue(Attribute metaValue) : storage(metaValue) {
    assert(isa<TypedAttr>(metaValue));
  }
  AnyValue(TypedAttr metaValue) : storage(RValue(metaValue)) {}
  AnyValue(Value rValue) : storage(RValue(rValue)) {}
  AnyValue(LValue lValue) : storage(lValue) {}

  bool isNull() const { return storage.isNull(); }
  bool operator!() const { return isNull(); }
  operator bool() const { return !isNull(); }

  /// If this contains a metavalue, return it; otherwise return null.
  TypedAttr getIfMValue() const {
    // Meta values are stored as Attribute because they are a single word, but
    // we know they always hold a TypedAttr.
    if (auto rvalue = dyn_cast<RValue>(storage))
      return rvalue.getIfMValue();
    return {};
  }

  RValue getIfRValue() const { return dyn_cast<RValue>(storage); }

  Value getIfDRValue() const {
    if (auto rvalue = getIfRValue())
      return rvalue.getIfDRValue();
    return {};
  }

  LValue getIfLValue() const { return dyn_cast<LValue>(storage); }

  /// Return the type for the contained representation, or null if they are
  /// both null.  In the case of an LValue, this will return the PointerType.
  Type getType() const;

private:
  PointerUnion<RValue, LValue> storage;
};

//===----------------------------------------------------------------------===//
// ExprNode
//===----------------------------------------------------------------------===//

/// Base class for all expression nodes.  Note that these nodes are not allowed
/// to own memory since they are bump pointer allocated and their destructors
/// are never run.
class ExprNode {
public:
  // This indicates the subclass.
  enum Kind {
    kIntLiteral,    // 42
    kFloatLiteral,  // 1.1
    kStringLiteral, // "Hello"
    kNoneLiteral,   // None
    kDeclRef,       // x
    kAttributeRef,  // x.y
    kCall,          // thing(a, b)
    kSubscript,     // thing[a, b:c]
    kParenExprNode, // (x+y)

    // Unary expressions.
    kUnaryMinus,
    kUnaryPlus,
    kUnaryTilde,
    kUnaryAmp,
    kFirstUnaryOp = kUnaryMinus,
    klastUnaryOp = kUnaryAmp,

    // Binary expressions.
    kAdd,
    kSub,
    kMul,
    kDiv,
    kExp,
    kFirstBinOp = kAdd,
    kLastBinOp = kExp,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual AnyValue emitIR(ExprEmitter &state,
                          Type contextualType = {}) const = 0;

  /// Emit this expression tree to an MLIR type.  This returns null on error,
  /// unlike the corresponding ExprEmitter method.
  virtual std::pair<Type, ASTType> emitType(ExprEmitter &state) const = 0;
};

//===----------------------------------------------------------------------===//
// ExprEmitter
//===----------------------------------------------------------------------===//

class ExprEmitter {
public:
  /// This is the shared state for the parser overall.
  LitSharedState &shared;

  /// This is scope to resolve declaration references against.
  ASTDecl &declScope;

  /// This is the current builder to emit into if we are allowed to generate a
  /// value.  This will be None when in a context that only allows parameters.
  /// It is mutable to support expressions that require internal control flow.
  Optional<OpBuilder> builder;

  /// When non-null, implicitly declared variables are added above this
  /// location.
  Operation *varDeclCursor;

  ExprEmitter(LitSharedState &shared, ASTDecl &declScope,
              Optional<OpBuilder> builder, Operation *varDeclCursor)
      : shared(shared), declScope(declScope), builder(builder),
        varDeclCursor(varDeclCursor) {}

  MLIRContext *getContext() const { return shared.context; }

  /// This helper emits the specified value rep as an RValue.
  RValue emitRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitRValue(node->emitIR(*this), node->getLoc());
  }
  RValue emitRValue(AnyValue rep, SMLoc loc);

  /// This helper emits the specified value rep as a DRValue which has an SSA
  /// value representation, materializing MValues and loading LValues as
  /// needed.  This returns null if emission fails.
  Value emitDRValue(RValue rep, SMLoc loc);
  Value emitDRValue(AnyValue rep, SMLoc loc) {
    return emitDRValue(emitRValue(rep, loc), loc);
  }

  /// This helper emits the specified value rep as an DRValue, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  Value emitDRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitDRValue(node->emitIR(*this), node->getLoc());
  }

  /// This helper emits the specified expression as a meta value, diagnosing the
  /// problem if the expression is only valid as a runtime value (using the
  /// specified message).  This returns null if emission fails.
  TypedAttr emitMValue(const ExprNode *node, const Twine &message);

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will
  /// that that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  LValue emitLValue(const ExprNode *node, Type contextualType,
                    const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null MLIR Types - if the
  /// expression is erroneous, it is diagnosed and a TypeCheckErrorType is
  /// returned, along with an erroneous AST type.
  std::pair<Type, ASTType> emitType(const ExprNode *node);

  /// Perform a name lookup in the current scope and return the named
  /// declaration.  This emits an error and returns null on error.
  ASTDecl *lookupDecl(StringRef name, SMLoc loc);

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

} // namespace M::KGEN::LIT

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::RValue> {
public:
  using RValue = M::KGEN::LIT::RValue;
  using Impl = PointerUnion<mlir::Attribute, mlir::Value>;
  using ImplTraits = PointerLikeTypeTraits<Impl>;
  static inline void *getAsVoidPointer(RValue value) {
    return const_cast<void *>(ImplTraits::getAsVoidPointer(value.getStorage()));
  }
  static inline RValue getFromVoidPointer(void *pointer) {
    return RValue(ImplTraits::getFromVoidPointer(pointer));
  }
  enum {
    NumLowBitsAvailable = PointerLikeTypeTraits<Impl>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::LValue> {
public:
  using LValue = M::KGEN::LIT::LValue;
  static inline void *getAsVoidPointer(LValue value) {
    return const_cast<void *>(value.getAsOpaquePointer());
  }
  static inline LValue getFromVoidPointer(void *pointer) {
    return LValue(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};
} // namespace llvm
#endif // LIT_EXPRS_H
