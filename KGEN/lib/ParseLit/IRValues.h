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
//   AnyValue               <- Expr emitted to MLIR
//     LValue (Value)       <- Expr with a runtime address.
//     RValue               <- Expr without an address
//       DRValue (Value)    <- Expr with a dynamic value (only known at runtime)
//       MValue (TypedAttr) <- Expr with a meta-value (known at compile time)
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

/// Instances of DRValue model a dynamic value represented with an SSA value.
/// It is described with an explicit type to clarify what sort of value it is,
/// differentiating it from an emitted LValue.  This helps avoid subtle bugs in
/// the emission phase.
class DRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  DRValue(Value v) : Value(v) {}
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
};

/// Instances of MValue model compile time values that are represented as MLIR
/// attributes.
class MValue {
public:
  MValue() {}
  MValue(TypedAttr v) : storage(v) {}
  MValue(Type value);

  MValue &operator=(TypedAttr newVal) {
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

  /// If this value /is/ a type return it.
  /// FIXME: virtually all users of this are going to be incorrect with type
  /// variables.
  Type getIfTypeValue() const;

  const void *getAsOpaquePointer() const {
    return storage.getAsOpaquePointer();
  }

  static MValue getFromOpaquePointer(void *ptr) {
    return {Attribute::getFromOpaquePointer(ptr)};
  }

private:
  Attribute storage;
};
raw_ostream &operator<<(raw_ostream &os, MValue value);
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, MValue value);

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
struct PointerLikeTypeTraits<M::KGEN::LIT::MValue> {
public:
  using MValue = M::KGEN::LIT::MValue;
  static const void *getAsVoidPointer(MValue value) {
    return value.get().getAsOpaquePointer();
  }
  static MValue getFromVoidPointer(void *pointer) {
    return MValue(cast_or_null<mlir::TypedAttr>(
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
  using Storage = PointerUnion<MValue, DRValue, LValue>;

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

/// RValue = MValue|DRValue.
class RValue : public VariantValueStorage<RValue> {
public:
  RValue() {}
  RValue(MValue metaValue) : VariantValueStorage(metaValue) {}
  RValue(TypedAttr value) : VariantValueStorage(value) {}
  RValue(Type value) : VariantValueStorage(MValue(value)) {}
  RValue(DRValue value) : VariantValueStorage(value) {}

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<MValue, DRValue>(storage))
      result.storage = storage;
    return result;
  }

  /// If this contains a metavalue, return it; otherwise return null.
  MValue getIfMValue() const { return dyn_cast<MValue>(storage); }
  DRValue getIfDRValue() const { return dyn_cast<DRValue>(storage); }

  /// If this value /is/ a type return it.
  /// FIXME: virtually all users of this are going to be incorrect with type
  /// variables.
  Type getIfTypeValue() const {
    if (auto mValue = getIfMValue())
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
  AnyValue(MValue value) : VariantValueStorage(value) {}
  AnyValue(RValue value) : VariantValueStorage(value.getStorage()) {}
  AnyValue(TypedAttr value) : VariantValueStorage(MValue(value)) {}
  AnyValue(Type value) : VariantValueStorage(MValue(value)) {}
  AnyValue(DRValue value) : VariantValueStorage(value) {}
  AnyValue(LValue value) : VariantValueStorage(value) {}

  LValue getIfLValue() const { return dyn_cast<LValue>(storage); }
  DRValue getIfDRValue() const { return dyn_cast<DRValue>(storage); }
  MValue getIfMValue() const { return dyn_cast<MValue>(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }

  /// If this value /is/ a type return it.
  /// FIXME: virtually all users of this are going to be incorrect with type
  /// variables.
  Type getIfTypeValue() const {
    if (auto mValue = getIfMValue())
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
