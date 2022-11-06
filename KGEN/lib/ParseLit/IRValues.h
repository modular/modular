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
//       MValue             <- Expr with a meta-value (known at compile time)
//         MAValue          <- MLIR Attribute type value
//         ASTType          <- Type value
//
//===----------------------------------------------------------------------===//

#ifndef IRVALUES_H
#define IRVALUES_H

#include "ASTType.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerUnion.h"

namespace M::KGEN::LIT {

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
  ASTType getIfMTValue() const { return dyn_cast<ASTType>(storage); }

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
  AnyValue(ASTType value) : VariantValueStorage(value) {}
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

#endif // IRVALUES_H
