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
//     LValue (Value)       <- Expr with a runtime address.
//     RValue               <- Expr without an address
//       DRValue (Value)    <- Expr with a dynamic value (only known at runtime)
//       MValue             <- Expr with a meta-value (known at compile time)
//         MAValue          <- MLIR Attribute type value
//         ASTType          <- Type value
//
//===----------------------------------------------------------------------===//

#ifndef LIT_EXPRS_H
#define LIT_EXPRS_H

#include "ASTType.h"
#include "LitSharedState.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {
using llvm::SMLoc;
class ExprEmitter;
class ASTDecl;

template <typename ValueType>
struct ASTTypeAnd {
  ValueType ir; // This is the IR representation of this.
  ASTType type; // This is the AST type.

  bool isNull() const { return ir; }
  bool operator!() const { return !ir; }
  operator bool() const { return ir; }

  FullType getFullType() const { return {ir.getType(), type}; }
};

/// Instances of DRValue model a dynamic value represented with an SSA value.
/// It is described with an explicit type to clarify what sort of value it is,
/// differentiating it from an emitted LValue.  This helps avoid subtle bugs in
/// the emission phase.
struct DRValue : public Value {
  using Value::Value;
  using Value::operator=;
  DRValue(Value v) : Value(v) {}
};

/// Instances of LValue model a dynamic address, which will always have pointer
/// type.  It is described with an explicit type because the type of the
/// underlying MLIR value may be pointer type for RValues of pointer type, so we
/// need something explicit to represent this.  This also helps avoid subtle
/// bugs in the emission phase.
struct LValue : public Value {
  using Value::Value;
  LValue(Value v) : Value(v) {}
};

/// Instances of MAValue model compile time values that are represented as MLIR
/// attributes.
struct MAValue {
  MAValue() {}
  MAValue(TypedAttr v) : storage(v) {}

  MAValue &operator=(TypedAttr newVal) {
    storage = newVal;
    return *this;
  }

  bool isNull() const { return storage == Attribute(); }
  bool operator!() const { return isNull(); }
  operator bool() const { return !isNull(); }

  TypedAttr get() const { return cast_or_null<TypedAttr>(storage); }
  operator TypedAttr() const { return get(); }

  /// Return the type for the contained representation, or null if null.
  Type getType() const { return get().getType(); }

  const void *getAsOpaquePointer() const {
    return storage.getAsOpaquePointer();
  }

  static MAValue getFromOpaquePointer(void *ptr) {
    MAValue result;
    result.storage = Attribute::getFromOpaquePointer(ptr);
    return result;
  }

private:
  Attribute storage;
};

} // namespace M::KGEN::LIT

namespace llvm {

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::LValue> {
public:
  using LValue = M::KGEN::LIT::LValue;
  static const void *getAsVoidPointer(LValue value) {
    return value.getAsOpaquePointer();
  }
  static LValue getFromVoidPointer(void *pointer) {
    return LValue(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::DRValue> {
public:
  using DRValue = M::KGEN::LIT::DRValue;
  static void *getAsVoidPointer(DRValue value) {
    return value.getAsOpaquePointer();
  }
  static DRValue getFromVoidPointer(void *pointer) {
    return DRValue(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MAValue> {
public:
  using MAValue = M::KGEN::LIT::MAValue;
  static const void *getAsVoidPointer(MAValue value) {
    return value.getAsOpaquePointer();
  }
  static MAValue getFromVoidPointer(void *pointer) {
    return MAValue(cast_or_null<mlir::TypedAttr>(
        mlir::Attribute::getFromOpaquePointer(pointer)));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};
} // namespace llvm

namespace M::KGEN::LIT {

template <typename DerivedType>
struct VariantValueStorage {
  /// These are all the forms of storage we can have.
  using Storage = PointerUnion<MAValue, ASTType, DRValue, LValue>;

  bool isNull() const { return storage.isNull(); }
  bool operator!() const { return storage.isNull(); }
  operator bool() const { return !storage.isNull(); }

  static DerivedType getFromStorage(Storage storage) {
    DerivedType result;
    result.storage = storage;
    return result;
  }

  Storage getStorage() const { return storage; }

protected:
  VariantValueStorage() {} // All are default constructible.
  template <typename T>
  VariantValueStorage(T init) : storage(init) {}

  Storage storage;
};

/// Instances of MValue model a known-compile-time meta value, represented by an
/// MLIR attribute or an ASTType.
///
/// MValue = MAValue|ASTType.
struct MValue : public VariantValueStorage<MValue> {
  MValue() {}
  MValue(TypedAttr attr) : VariantValueStorage(MAValue(attr)) {}
  MValue(MAValue value) : VariantValueStorage(value) {}
  MValue(ASTType type) : VariantValueStorage(type) {}

  static MValue getFrom(Storage storage) {
    // Initialize conditionally based on what is in Storage.
    MValue result;
    if (isa<MAValue, ASTType>(storage))
      result.storage = storage;
    return result;
  }

  MAValue getIfMAValue() const { return dyn_cast<MAValue>(storage); }
  ASTType getIfType() const { return dyn_cast<ASTType>(storage); }

  /// Return the type for the contained representation, or null if null.
  Type getType(MLIRContext *context) const;
};

/// RValue = MValue|DRValue.
class RValue : public VariantValueStorage<RValue> {
public:
  RValue() {}
  RValue(MValue metaValue) : VariantValueStorage(metaValue.getStorage()) {}
  RValue(TypedAttr value) : VariantValueStorage(value) {}
  RValue(DRValue value) : VariantValueStorage(value) {}

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<MAValue, ASTType, DRValue>(storage))
      result.storage = storage;
    return result;
  }

  /// If this contains a metavalue, return it; otherwise return null.
  MValue getIfMValue() const { return MValue::getFrom(storage); }
  DRValue getIfDRValue() const { return dyn_cast<DRValue>(storage); }
  MAValue getIfMAValue() const { return dyn_cast<MAValue>(storage); }
  ASTType getIfMTValue() const { return dyn_cast<ASTType>(storage); }

  /// Return the type for the contained representation, or null if null.
  Type getType(MLIRContext *context) const;
};

/// AnyValue = RValue|LValue.
class AnyValue : public VariantValueStorage<RValue> {
public:
  AnyValue() {}
  AnyValue(MValue value) : VariantValueStorage(value.getStorage()) {}
  AnyValue(RValue value) : VariantValueStorage(value.getStorage()) {}
  AnyValue(TypedAttr value) : VariantValueStorage(MAValue(value)) {}
  AnyValue(MAValue value) : VariantValueStorage(value) {}
  AnyValue(DRValue value) : VariantValueStorage(value) {}
  AnyValue(LValue value) : VariantValueStorage(value) {}

  LValue getIfLValue() const { return dyn_cast<LValue>(storage); }
  DRValue getIfDRValue() const { return dyn_cast<DRValue>(storage); }
  MAValue getIfMAValue() const { return dyn_cast<MAValue>(storage); }
  ASTType getIfMTValue() const { return dyn_cast<ASTType>(storage); }

  RValue getIfRValue() const { return RValue::getFrom(storage); }
  MValue getIfMValue() const { return MValue::getFrom(storage); }

  /// Return the type for the contained representation, or null if they are
  /// both null.  In the case of an LValue, this will return the PointerType.
  Type getType(MLIRContext *context) const;
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
    kMatrixMul,
    kDiv,
    kFloorDiv,
    kModulo,
    kBoolOr,
    kBoolAnd,
    kBoolNot,
    kCmpIn,
    kCmpNotIn,
    kCmpIs,
    kCmpIsNot,
    kCmpLess,
    kCmpLessEqual,
    kCmpGreater,
    kCmpGreaterEqual,
    kCmpNotEqual,
    kCmpEqual,
    kBitwiseOr,
    kBitwiseXor,
    kBitwiseAnd,
    kLeftShift,
    kRightShift,
    kExp,
    kFirstBinOp = kAdd,
    kLastBinOp = kExp,

    // Ternary expressions.
    kIfElse,
  } const kind;

  ExprNode(Kind kind) : kind(kind) {}
  virtual ~ExprNode();

  /// Return the primary location for this node for error reporting purposes.
  virtual SMLoc getLoc() const = 0;

  /// Emit this expression to MLIR, returning a (possibly null!) AnyValue.  The
  /// contextualType (if non-null) indicates the contextual type to use for an
  /// implicitly declared value, e.g. a/b in `def f(): (a,b) = (1,2)`.
  virtual ASTTypeAnd<AnyValue> emitIR(ExprEmitter &state,
                                      FullType contextualType = {}) const = 0;

  /// Emit this expression tree to an MLIR type.  This returns null on error,
  /// unlike the corresponding ExprEmitter method.
  virtual FullType emitType(ExprEmitter &state) const = 0;
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
  ASTTypeAnd<RValue> emitRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitRValue(node->emitIR(*this), node->getLoc());
  }
  ASTTypeAnd<RValue> emitRValue(ASTTypeAnd<AnyValue> rep, SMLoc loc);

  /// This helper emits the specified value rep as a DRValue which has an SSA
  /// value representation, materializing MValues and loading LValues as
  /// needed.  This returns null if emission fails.
  ASTTypeAnd<DRValue> emitDRValue(ASTTypeAnd<RValue> rep, SMLoc loc);
  ASTTypeAnd<DRValue> emitDRValue(ASTTypeAnd<AnyValue> rep, SMLoc loc) {
    return emitDRValue(emitRValue(rep, loc), loc);
  }

  /// This helper emits the specified value rep as an DRValue, materializing
  /// it as a parameter constant if it is a parameter.  This returns null if
  /// emission fails.
  ASTTypeAnd<DRValue> emitDRValue(const ExprNode *node) {
    assert(node && "cannot emit a null node");
    return emitDRValue(node->emitIR(*this), node->getLoc());
  }

  /// This helper emits the specified expression as a meta value, diagnosing the
  /// problem if the expression is only valid as a runtime value (using the
  /// specified message).  This returns null if emission fails.
  ASTTypeAnd<MValue> emitMValue(const ExprNode *node, const Twine &message);
  ASTTypeAnd<MAValue> emitMAValue(const ExprNode *node, const Twine &message);

  /// Emit the specified expression as an LValue which can be loaded and stored.
  /// If contextualType is non-null, then an implicitly declared LValue will
  /// that that type.
  ///
  /// This diagnoses the expression with the specified message if it isn't a
  /// valid LValue.
  ASTTypeAnd<LValue> emitLValue(const ExprNode *node, FullType contextualType,
                                const Twine &message);

  /// This helper emits the specified expression tree as a type, e.g. turning
  /// "Int" into the type for it.  This never returns null MLIR Types - if the
  /// expression is erroneous, it is diagnosed and a TypeCheckErrorType is
  /// returned, along with an erroneous AST type.
  FullType emitType(const ExprNode *node);

  /// Perform a name lookup in the current scope and return the named
  /// declaration.  This emits an error and returns null on error.
  ASTDecl *lookupDecl(StringRef name, SMLoc loc, ASTDecl &scope,
                      Twine errorMessage);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &twine) const {
    return shared.emitError(loc, twine);
  }

  /// Translate an SMLoc into an MLIR Location.
  Location translateLocation(SMLoc loc) const {
    return shared.translateLocation(loc);
  }
};

} // namespace M::KGEN::LIT

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MValue> {
public:
  using MValue = M::KGEN::LIT::MValue;
  using Storage = MValue::Storage;
  using StorageTraits = PointerLikeTypeTraits<Storage>;
  static void *getAsVoidPointer(MValue value) {
    return StorageTraits::getAsVoidPointer(value.getStorage());
  }
  static MValue getFromVoidPointer(void *pointer) {
    return MValue::getFromStorage(StorageTraits::getFromVoidPointer(pointer));
  }
  enum { NumLowBitsAvailable = StorageTraits::NumLowBitsAvailable };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::RValue> {
public:
  using RValue = M::KGEN::LIT::RValue;
  using Storage = RValue::Storage;
  using StorageTraits = PointerLikeTypeTraits<Storage>;
  static void *getAsVoidPointer(RValue value) {
    return StorageTraits::getAsVoidPointer(value.getStorage());
  }
  static RValue getFromVoidPointer(void *pointer) {
    return RValue::getFromStorage(StorageTraits::getFromVoidPointer(pointer));
  }
  enum { NumLowBitsAvailable = StorageTraits::NumLowBitsAvailable };
};
} // namespace llvm

#endif // LIT_EXPRS_H
