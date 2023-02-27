//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the representations used for expressions emitted to MLIR,
// either as operations for runtime values, or as attributes / ASTTypes for
// metavalues.
//
// Emitting an expression to MLIR can either produce a meta-value as an rvalue,
// may produce a runtime value as an rvalue, or may produce a metavalue as an
// LValue.  These make up the following hierarchy of value kinds:
//
//   AnyValue           <- Expr emitted to MLIR:
//     LValue (Value)     <- ... with a runtime address.
//     RValue             <- ... as an independent value
//       SRValue (Value)     <- ..with a dynamic SSA value
//       PRValue (TypedAttr) <- ..with a parameter value (known at compile time)
//
//===----------------------------------------------------------------------===//

#ifndef IRVALUES_H
#define IRVALUES_H

#include "ASTType.h"
#include "LitSharedState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"

namespace M::KGEN::LIT {
class ExprNode;

template <typename ValueType>
struct ASTExprAnd {
  ValueType ir;

  /// This is the expression a value was produced from, carrying location and
  /// additional semantic information.
  const ExprNode *expr;

  bool isNull() const { return ir.isNull(); }
  bool operator!() const { return !ir; }
  operator bool() const { return bool(ir); }

  template <typename OtherValueType>
  operator ASTExprAnd<OtherValueType>() const {
    return {OtherValueType(ir), expr};
  }
};

/// Instances of SRValue model a dynamic value represented with an SSA value.
/// It is described with an explicit type to clarify what sort of value it is,
/// differentiating it from an emitted LValue.  This helps avoid subtle bugs in
/// the emission phase.
class SRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SRValue(Value v) : Value(v) {}
};

/// Instances of LValue model a dynamic address, which will always have pointer
/// type.  It is described with an explicit C++ type so the expression emission
/// logic can better reason about it.
///
/// When produced by the emitter, the ASTType of an LValue is always the type
/// of dereferencing the LValue, there is no extra level of pointer added.
///
class LValue : public Value {
public:
  using Value::Value;
  LValue(Value v) : Value(v) {}

  /// This method returns the type of this value when projected as an RValue.
  /// If this is already an RValue, it is the type of the value.  If this is
  /// an LValue, it strips off the pointer type.
  ASTType getRValueType() const;
};

/// Instances of PRValue model compile time values that are represented as MLIR
/// attributes.
class PRValue {
public:
  PRValue() {}
  PRValue(TypedAttr v) : storage(v) {}
  PRValue(Type value);

  PRValue &operator=(TypedAttr newVal) {
    storage = newVal;
    return *this;
  }

  bool isNull() const { return storage == Attribute(); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  TypedAttr get() const { return cast_or_null<TypedAttr>(storage); }
  operator TypedAttr() const { return get(); }

  /// Return the type for the contained representation, or null if null.
  Type getType() const { return get().getType(); }

  /// If this value /is/ a type (i.e., if it has metatype type) return it.
  ASTType getIfTypeValue() const;

  const void *getAsOpaquePointer() const {
    return storage.getAsOpaquePointer();
  }

  static PRValue getFromOpaquePointer(void *ptr) {
    return {Attribute::getFromOpaquePointer(ptr)};
  }

private:
  Attribute storage;
};
raw_ostream &operator<<(raw_ostream &os, PRValue value);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, PRValue value);

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
struct PointerLikeTypeTraits<M::KGEN::LIT::SRValue> {
public:
  using SRValue = M::KGEN::LIT::SRValue;
  static void *getAsVoidPointer(SRValue value) {
    return value.getAsOpaquePointer();
  }
  static SRValue getFromVoidPointer(void *pointer) {
    return SRValue(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::PRValue> {
public:
  using PRValue = M::KGEN::LIT::PRValue;
  static const void *getAsVoidPointer(PRValue value) {
    return value.get().getAsOpaquePointer();
  }
  static PRValue getFromVoidPointer(void *pointer) {
    return PRValue(cast_or_null<mlir::TypedAttr>(
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
  using Storage = PointerUnion<PRValue, SRValue, LValue>;

  bool isNull() const { return storage.isNull(); }
  bool operator!() const { return storage.isNull(); }
  explicit operator bool() const { return !storage.isNull(); }

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

/// RValue = PRValue|SRValue.
class RValue : public VariantValueStorage<RValue> {
public:
  RValue() {}
  RValue(PRValue metaValue) : VariantValueStorage(metaValue) {}
  RValue(TypedAttr value) : VariantValueStorage(value) {}
  RValue(Type value) : VariantValueStorage(PRValue(value)) {}
  RValue(SRValue value) : VariantValueStorage(value) {}

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa_and_nonnull<PRValue, SRValue>(storage))
      result.storage = storage;
    return result;
  }

  /// If this contains a metavalue, return it; otherwise return null.
  PRValue getIfPRValue() const { return dyn_cast<PRValue>(storage); }
  SRValue getIfSRValue() const { return dyn_cast<SRValue>(storage); }

  /// If this value /is/ a type return it.
  ASTType getIfTypeValue() const {
    if (auto mValue = getIfPRValue())
      return mValue.getIfTypeValue();
    return {};
  }

  /// Return the type for the contained representation, or null if null.
  Type getType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, RValue value);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, RValue value);

/// AnyValue = RValue|LValue.
class AnyValue : public VariantValueStorage<AnyValue> {
public:
  AnyValue() {}
  AnyValue(PRValue value) : VariantValueStorage(value) {}
  AnyValue(RValue value) : VariantValueStorage(value.getStorage()) {}
  AnyValue(TypedAttr value) : VariantValueStorage(PRValue(value)) {}
  AnyValue(Type value) : VariantValueStorage(PRValue(value)) {}
  AnyValue(SRValue value) : VariantValueStorage(value) {}
  AnyValue(LValue value) : VariantValueStorage(value) {}

  LValue getIfLValue() const { return dyn_cast_or_null<LValue>(storage); }
  SRValue getIfSRValue() const { return dyn_cast_or_null<SRValue>(storage); }
  PRValue getIfPRValue() const { return dyn_cast_or_null<PRValue>(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }

  /// This method returns the type of this value when projected as an RValue.
  /// If this is already an RValue, it is the type of the value.  If this is
  /// an LValue, it strips off the pointer type.
  ASTType getRValueType() const;

  /// If this value /is/ a type return it.
  ASTType getIfTypeValue() const {
    if (auto mValue = getIfPRValue())
      return mValue.getIfTypeValue();
    return {};
  }

  /// Return the type for the contained representation, or null if they are
  /// both null.
  Type getType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, AnyValue value);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, AnyValue value);

} // namespace M::KGEN::LIT

namespace llvm {
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

#endif // IRVALUES_H
