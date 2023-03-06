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
// AnyValue              <- Expr emitted to MLIR:
//   LValue (Value)        <- ... with a runtime address.
//   RValue                <- ... as an independent value
//     ORValue (OverloadSet) <- ..with an unresolved overload set
//     CRValue               <- ..with a resolved type
//       SRValue (Value)     <- ..with a dynamic value loaded into SSA register
//       MRValue (Value)     <- ..with a dynamic value emitted into memory
//       PRValue (TypedAttr) <- ..with a parameter value (known at compile time)
//
// Note that SRValue is not compatible with memory-primary types, but MRValue
// can hold any type, including a register compatible type.
//
//===----------------------------------------------------------------------===//

#ifndef IRVALUES_H
#define IRVALUES_H

#include "ASTType.h"
#include "LitSharedState.h"

#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ADT/SmartVariant.h"
#include "mlir/IR/Value.h"

namespace M::KGEN::LIT {
class ExprNode;
class ExprEmitter;
class OverloadSet;
class FuncOp;
class ValueDest;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

template <typename ValueType>
struct ASTExprAnd {
  ValueType ir;

  /// This is the expression a value was produced from, carrying location and
  /// additional semantic information.
  const ExprNode *expr;

  bool isNull() const { return ir.isNull(); }
  bool operator!() const { return !ir; }
  explicit operator bool() const { return bool(ir); }

  template <typename OtherValueType>
  operator ASTExprAnd<OtherValueType>() const {
    return {OtherValueType(ir), expr};
  }
};

//===----------------------------------------------------------------------===//
// Value Classifications
//===----------------------------------------------------------------------===//

/// This is used to provide a null representation for the SmartVariant, allowing
/// it to be default constructed to a known state.
class NullRepresentation {};

/// Instances of SRValue model a dynamic value loaded into an SSA value.  This
/// representation can only be used with register-primary types.
class SRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SRValue(Value v) : Value(v) {}

  ASTType getType() const { return ASTType(Value::getType()); }
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

  ASTType getType() const { return ASTType(Value::getType()); }
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

  ASTType getType() const { return ASTType(Value::getType()); }
};

/// Instances of PRValue model compile time values that are represented as MLIR
/// attributes.
class PRValue {
public:
  PRValue() {}
  PRValue(TypedAttr v) : storage(v) {}
  PRValue(Attribute value) : storage(value) {
    assert(isa<TypedAttr>(value) && "invalid value attribute");
  }

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
  ASTType getType() const { return get().getType(); }

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

/// Instances of ORValue represent an unresolved overload set that must be
/// disambiguated before being used.
class ORValue {
public:
  ORValue();
  ORValue(const ORValue &existing);
  ORValue &operator=(const ORValue &existing);
  ~ORValue();

  bool isNull() const { return !storage; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  OverloadSet *operator->();
  const OverloadSet *operator->() const;

  template <typename... Args>
  static ORValue create(Args &&...args);
  static ORValue create(OverloadSet &&set);

private:
  struct OverloadSetWrapper;
  ORValue(LLCL::RCRef<OverloadSetWrapper> storage);

  LLCL::RCRef<OverloadSetWrapper> storage;
};
raw_ostream &operator<<(raw_ostream &os, ORValue value);

template <typename DerivedType>
struct VariantValueStorage {
  /// These are all the forms of storage we can have.
  using Storage = SmartVariant<NullRepresentation, PRValue, SRValue, MRValue,
                               ORValue, LValue>;

  VariantValueStorage()
      : storage(NullRepresentation()) {} // All are default constructible.

  // These are common constructors all VariantValueStorage's have.
  VariantValueStorage(PRValue value) {
    if (value)
      storage = value;
  }
  VariantValueStorage(TypedAttr value) : VariantValueStorage(PRValue(value)) {}
  VariantValueStorage(Attribute value) : VariantValueStorage(TypedAttr(value)) {
    assert(isa<TypedAttr>(value) && "invalid value attribute");
  }
  VariantValueStorage(Type value) : VariantValueStorage(PRValue(value)) {}
  VariantValueStorage(ASTType value) : VariantValueStorage(PRValue(value)) {}
  VariantValueStorage(SRValue value) {
    if (value)
      storage = value;
  }
  VariantValueStorage(MRValue value) {
    if (value)
      storage = value;
  }

  PRValue getIfPRValue() const { return dyn_cast<PRValue>(storage); }
  SRValue getIfSRValue() const { return dyn_cast<SRValue>(storage); }
  MRValue getIfMRValue() const { return dyn_cast<MRValue>(storage); }

  /// If this value is a PRValue for a type, then return the type.
  ASTType getIfTypeValue() const {
    if (auto mValue = getIfPRValue())
      return mValue.getIfTypeValue();
    return {};
  }

  bool isNull() const { return isa<NullRepresentation>(storage); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  static DerivedType getFromStorage(Storage storage) {
    DerivedType result;
    result.storage = std::move(storage);
    return result;
  }

  Storage getStorage() const { return storage; }

protected:
  Storage storage;
};

/// Concrete RValue: CRValue = PRValue|SRValue|MRValue.
class CRValue : public VariantValueStorage<CRValue> {
public:
public:
  using VariantValueStorage::VariantValueStorage;

  static CRValue getFrom(Storage storage) {
    CRValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa_and_nonnull<PRValue, SRValue, MRValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in a MRValue to return the
  /// underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, CRValue value);

/// RValue = CRValue|ORValue.
class RValue : public VariantValueStorage<RValue> {
public:
  RValue() {}
  RValue(PRValue value) {
    if (value)
      storage = value;
  }
  RValue(TypedAttr value) : RValue(PRValue(value)) {}
  RValue(Type value) : RValue(PRValue(value)) {}
  RValue(SRValue value) {
    if (value)
      storage = value;
  }
  RValue(MRValue value) {
    if (value)
      storage = value;
  }
  RValue(ORValue value) {
    if (value)
      storage = std::move(value);
  }

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa_and_nonnull<PRValue, SRValue, MRValue, ORValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  CRValue getIfCRValue() const { return CRValue::getFrom(storage); }
  ORValue getIfORValue() const { return dyn_cast<ORValue>(storage); }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, RValue value);

/// AnyValue = RValue|LValue.
class AnyValue : public VariantValueStorage<AnyValue> {
public:
  using VariantValueStorage::VariantValueStorage;
  AnyValue(ORValue value) {
    if (value)
      storage = std::move(value);
  }
  AnyValue(CRValue value) { storage = value.getStorage(); }
  AnyValue(RValue value) { storage = value.getStorage(); }
  AnyValue(LValue value) { storage = value; }

  LValue getIfLValue() const { return dyn_cast<LValue>(storage); }
  ORValue getIfORValue() const { return dyn_cast<ORValue>(storage); }
  CRValue getIfCRValue() const { return CRValue::getFrom(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }

  /// This method returns the type of this value when projected as an RValue.
  /// If this is already an RValue, it is the type of the value.  If this is
  /// an LValue, it strips off the pointer type.
  ASTType getRValueType() const;

  /// Return the type for the contained representation, or null if they are
  /// both null.
  ASTType getType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, AnyValue value);

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

#endif // IRVALUES_H
