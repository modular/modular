//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the representations used for expressions emitted to MLIR,
// either as MLIR SSA values for runtime values and addresses, or as MLIR
// attributes for parameter values.
//
// Emitting an expression to MLIR may produce one of the follow representations,
// from a hierarchy of value kinds:
//
//   AnyValue           <- Expr emitted to MLIR:
//     LValue (Value)     <- ... with a runtime address.
//     RValue             <- ... as an independent value
//       SRValue (Value)     <- ..with a dynamic value loaded into SSA register
//       MRvalue (Value)     <- ..with a dynamic value emitted into memory
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

/// Instances of SRValue model a dynamic value loaded into an SSA value.  This
/// representation can only be used with register-primary types.
class SRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SRValue(Value v) : Value(v) {}
};

/// Instances of MRValue model a dynamic value stored into memory whose address
/// is represented with an SSA value.  This representation is typically used
/// with memory-primary types, but may also be used with register-primary types,
/// (e.g.) when initializing a var declaration.
class MRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MRValue(Value v) : Value(v) {}

  /// MRValue's represent the address of the stored value.  This returns the
  /// RValue type, the declared type of the value.
  ASTType getRValueType() const;
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

template <typename ActualType>
struct MLIRValueWrapper {
  static const void *getAsVoidPointer(ActualType value) {
    return value.getAsOpaquePointer();
  }
  static ActualType getFromVoidPointer(void *pointer) {
    return ActualType(mlir::Value::getFromOpaquePointer(pointer));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Value>::NumLowBitsAvailable
  };
};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::LValue>
    : public MLIRValueWrapper<M::KGEN::LIT::LValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::SRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::SRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MRValue> {};

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
        PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable
  };
};
} // namespace llvm

namespace M::KGEN::LIT {

template <typename DerivedType>
struct VariantValueStorage {
  /// These are all the forms of storage we can have.
  using Storage = PointerUnion<PRValue, SRValue, MRValue, LValue>;

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
  RValue(MRValue value) : VariantValueStorage(value) {}

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa_and_nonnull<PRValue, SRValue, MRValue>(storage))
      result.storage = storage;
    return result;
  }

  /// If this contains a metavalue, return it; otherwise return null.
  PRValue getIfPRValue() const { return dyn_cast_or_null<PRValue>(storage); }
  SRValue getIfSRValue() const { return dyn_cast_or_null<SRValue>(storage); }
  MRValue getIfMRValue() const { return dyn_cast_or_null<MRValue>(storage); }

  /// If this value is a PRValue for a type, then return the type.
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
  AnyValue(MRValue value) : VariantValueStorage(value) {}
  AnyValue(LValue value) : VariantValueStorage(value) {}

  LValue getIfLValue() const { return dyn_cast_or_null<LValue>(storage); }
  SRValue getIfSRValue() const { return dyn_cast_or_null<SRValue>(storage); }
  MRValue getIfMRValue() const { return dyn_cast_or_null<MRValue>(storage); }
  PRValue getIfPRValue() const { return dyn_cast_or_null<PRValue>(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }

  /// This method returns the type of this value when projected as an RValue.
  /// If this is already an RValue, it is the type of the value.  If this is
  /// an LValue, it strips off the pointer type.
  ASTType getRValueType() const;

  /// If this value is a PRValue for a type, then return the type.
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
