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
// which can be classified in this type level hierarchy (but notice that PValue
// is both a BValue and RValue):
//
// AnyValue       <- Expr emitted to MLIR...
//   LValue         <- mutable reference to storage
//     XLValue        <- value is in memory with a mutable reference
//     MLValue        <- value is in memory
//     DLValue        <- with dynamic get/set accessors
//   BValue         <- with a borrowed value
//     SBValue          <- value is register-passable and in an SSA register
//     XBValue          <- value is in memory with a reference (may be mutable)
//     MBValue          <- value is in memory
//     PValue           <- value is a parameter expression.
//   RValue         <- with an owned value
//     URValue         <- value cannot be materialized
//       ORValue        <- with an unresolved overload set
//     CRValue        <- with a concrete resolved type
//       SRValue        <- with a register-passable value in an SSA register
//       XRValue        <- value is in memory with a mutable reference
//       MRValue        <- with an owned value in memory
//       PValue         <- with a parameter value
//
// This is another parallel hierarchy, which excludes URValue:
//
//   CValue        <- Concrete value: LValue or RValue with a known type.
//     LValue        <- mutable reference
//     BValue        <- Borrowed value
//     CRValue       <- Concrete RValue
//
// Note that SRValue is not compatible with memory-only types, but XRValue
// can hold any type, including a register compatible type.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_IRVALUES_H
#define KGEN_MOJOPARSER_IRVALUES_H

#include "KGEN/MojoParser/ASTType.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace M::KGEN::LIT {
class BaseDLValue;
class ExprNode;
class ExprEmitter;
class OverloadSet;
class FuncOp;
class ValueDest;
class StructFieldOp;
class GlobalVarDeclOp;

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
/// representation can only be used with register-passable types.
class SRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SRValue(Value v) : Value(v) {}

  ASTType getType() const { return ASTType(Value::getType()); }
};

/// Instances of MRValue model a dynamic value stored into memory whose address
/// is represented with an SSA value.  This representation is typically used
/// with memory-only types, but may also be used with register-passable types,
/// (e.g.) when initializing a var declaration.  Values of this type are owned
/// instances of a value that needs to be consumed, akin to an x-value in C++.
class MRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MRValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getPointerElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of XRValue model a dynamic value stored into memory whose address
/// is represented with an SSA value.  This representation is typically used
/// with memory-only types, but may also be used with register-passable types,
/// (e.g.) when initializing a var declaration.  Values of this type are owned
/// instances of a value that needs to be consumed, akin to an x-value in C++.
class XRValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  XRValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of XBValue model a borrowed reference to dynamic value stored
/// into memory. The address is represented with an SSA value of !lit.ref type,
/// which might (or might not) be mutable.
///
/// This representation is used for borrowed arguments, and for some expressions
/// like `a.b` where `a` is an XRValue or XBValue (like a let) and `b` is a
/// stored property.
class XBValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  XBValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of MBValue model a borrowed reference to dynamic value stored
/// into memory; the address is represented with an SSA value.  This
/// representation is used for borrowed arguments, and for some expressions
/// like `a.b` where `a` is an MRValue or MBValue (like a let) and `b` is a
/// stored property.
class MBValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  MBValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getPointerElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of SBValue model a borrowed reference to a dynamic value stored
/// in an SSA register.  This representation is used for borrowed arguments, and
/// for some expressions like `a.b` where `a` is an MRValue or MBValue (like a
/// let) and `b` is a stored property.
class SBValue : public Value {
public:
  using Value::Value;
  using Value::operator=;
  SBValue(Value v) : Value(v) {}

  ASTType getType() const { return ASTType(Value::getType()); }
};

/// Instances of MLValue model a loadable/storable address as an SSA value.  It
/// always has !kgen.pointer type.
class MLValue : public Value {
public:
  using Value::Value;
  MLValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getPointerElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// Instances of XLValue model a loadable/storable address as an SSA value with
/// a mutable !lit.ref reference type.
class XLValue : public Value {
public:
  using Value::Value;
  XLValue(Value v) : Value(v) { check(); }

  /// This returns the declared type of the value without the wrapping pointer.
  ASTType getRValueType() const { return getType().getReferenceElementType(); }
  ASTType getType() const { return ASTType(Value::getType()); }

private:
  void check() const;
};

/// DLValue's model a dynamic LValue which has a getter and setter.  Lit
/// supports two ways to spell this - with property access `a.x =`
/// and with subscript syntax `a[i,j] = `, invoking __getattr__/__setattr__ and
/// __getitem__ and __setitem__ respectively.
///
/// DLValues are allowed to be get-only, set-only, or get-set.
class DLValue {
public:
  DLValue() = default;
  DLValue(RCRef<BaseDLValue> storage) : storage(std::move(storage)) {}
  DLValue(const DLValue &existing) : storage(existing.storage.copy()) {}
  DLValue &operator=(const DLValue &existing);
  ~DLValue();

  bool isNull() const { return !storage; }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  BaseDLValue *operator->() const { return &*storage; }

private:
  RCRef<BaseDLValue> storage;
};

/// Instances of PValue model compile time values that are represented as MLIR
/// attributes.  It is both a BValue and an RValue, it may contain both
/// @register_passable and memory-only types.
class PValue {
public:
  PValue() = default;
  PValue(TypedAttr v) : storage(v) {}
  PValue(Attribute value) : storage(value) {
    assert((!value || isa<TypedAttr>(value)) && "invalid value attribute");
  }

  PValue(Type value);

  PValue &operator=(TypedAttr newVal) {
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
  ASTType getRValueType() const { return getType(); }

  /// If this value /is/ a type (i.e., if it has metatype type) return it.
  ASTType getIfTypeValue() const;

  const void *getAsOpaquePointer() const {
    return storage.getAsOpaquePointer();
  }

  static PValue getFromOpaquePointer(void *ptr) {
    return {Attribute::getFromOpaquePointer(ptr)};
  }

  /// Pretty print the pvalue for use by diagnostics and other high level
  /// situations.
  void printForDiag(raw_ostream &os) const;

  void dump() const;

private:
  Attribute storage;
};
raw_ostream &operator<<(raw_ostream &os, PValue value);

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

  OverloadSet *operator->() { return &**this; }
  const OverloadSet *operator->() const { return &**this; }
  const OverloadSet &operator*() const;
  OverloadSet &operator*();

  template <typename... Args>
  static ORValue create(Args &&...args);
  static ORValue create(OverloadSet &&set);

private:
  struct OverloadSetWrapper;
  ORValue(RCRef<OverloadSetWrapper> storage);

  RCRef<OverloadSetWrapper> storage;
};
raw_ostream &operator<<(raw_ostream &os, ORValue value);

template <typename DerivedType>
struct VariantValueStorage {
  /// These are all the forms of storage we can have.
  using Storage = SmartVariant<NullRepresentation, PValue, SRValue, MRValue,
                               XRValue, ORValue, SBValue, MBValue, XBValue,
                               DLValue, MLValue, XLValue>;

  VariantValueStorage()
      : storage(NullRepresentation()) {} // All are default constructible.

  bool isNull() const { return isa<NullRepresentation>(storage); }
  bool operator!() const { return isNull(); }
  explicit operator bool() const { return !isNull(); }

  static DerivedType getFromStorage(const Storage &storage) {
    DerivedType result;
    result.storage = storage;
    return result;
  }

  Storage &getStorage() { return storage; }
  const Storage &getStorage() const { return storage; }

  // Return true if this is one of the scalar representation.
  bool isSValue() const {
    return isa<SRValue>(storage) || isa<SBValue>(storage);
  }
  // Return true if this is one of the memory representation.
  bool isMValue() const {
    return isa<MBValue>(storage) || isa<MBValue>(storage) ||
           isa<MLValue>(storage);
  }
  // Return true if this is one of the reference representation.
  bool isXValue() const {
    return isa<XBValue>(storage) || isa<XBValue>(storage) ||
           isa<XLValue>(storage);
  }

protected:
  Storage storage;
};

template <typename DerivedType>
struct VariantCRValue {
  VariantCRValue() = default;
  // These are common constructors all CRValues have.
  VariantCRValue(PValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantCRValue(TypedAttr value) : VariantCRValue(PValue(value)) {}
  VariantCRValue(Attribute value) : VariantCRValue(TypedAttr(value)) {
    assert(isa<TypedAttr>(value) && "invalid value attribute");
  }
  VariantCRValue(Type value) : VariantCRValue(PValue(value)) {}
  VariantCRValue(ASTType value) : VariantCRValue(PValue(value)) {}
  VariantCRValue(SRValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantCRValue(MRValue value) {
    if (value)
      getStorageR() = value;
  }
  VariantCRValue(XRValue value) {
    if (value)
      getStorageR() = value;
  }

  PValue getIfPValue() const { return dyn_cast<PValue>(getStorageR()); }
  SRValue getIfSRValue() const { return dyn_cast<SRValue>(getStorageR()); }
  MRValue getIfMRValue() const { return dyn_cast<MRValue>(getStorageR()); }
  XRValue getIfXRValue() const { return dyn_cast<XRValue>(getStorageR()); }

  /// If this value is a PValue for a type, then return the type.
  ASTType getIfTypeValue() const {
    if (auto value = getIfPValue())
      return value.getIfTypeValue();
    return {};
  }

private:
  // These are named getStorageR instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageR() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageR() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// Concrete RValue: CRValue = PValue|SRValue|MRValue|XRValue.
class CRValue : public VariantValueStorage<CRValue>,
                public VariantCRValue<CRValue> {
public:
  using VariantCRValue::VariantCRValue;
  using VariantValueStorage::VariantValueStorage;

  static CRValue getFrom(Storage storage) {
    CRValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<PValue, SRValue, MRValue, XRValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in a MRValue to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, CRValue value);

/// This is the base class of any URValue parent, enabling implicit conversion
/// from and checked conversion to child value types.
template <typename DerivedType>
struct VariantURValue {
  VariantURValue() = default;
  // These are common constructors all URValues have.
  VariantURValue(ORValue value) {
    if (value)
      getStorageR() = std::move(value);
  }

  ORValue getIfORValue() const { return dyn_cast<ORValue>(getStorageR()); }

private:
  // These are named getStorageR instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageR() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageR() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// URValue = ORValue
class URValue : public VariantValueStorage<URValue>,
                public VariantURValue<URValue> {
public:
  using VariantURValue::VariantURValue;
  using VariantValueStorage::VariantValueStorage;

  static URValue getFrom(Storage storage) {
    URValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<ORValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, URValue value);

/// RValue = CRValue|URValue.
class RValue : public VariantValueStorage<RValue>,
               public VariantCRValue<RValue>,
               public VariantURValue<RValue> {
public:
  using VariantCRValue::VariantCRValue;
  using VariantURValue::VariantURValue;
  using VariantValueStorage::VariantValueStorage;

  RValue() = default;

  RValue(URValue value) {
    if (value)
      storage = std::move(value.getStorage());
  }
  RValue(CRValue value) {
    if (value)
      storage = std::move(value.getStorage());
  }

  static RValue getFrom(Storage storage) {
    RValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<PValue, SRValue, MRValue, XRValue, ORValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  CRValue getIfCRValue() const { return CRValue::getFrom(storage); }
  URValue getIfURValue() const { return URValue::getFrom(storage); }

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, RValue value);

template <typename DerivedType>
struct VariantLValue {
  VariantLValue() = default;
  VariantLValue(MLValue value) {
    if (value)
      getStorageL() = value;
  }
  VariantLValue(XLValue value) {
    if (value)
      getStorageL() = value;
  }
  VariantLValue(DLValue value) { getStorageL() = value; }

  MLValue getIfMLValue() const { return dyn_cast<MLValue>(getStorageL()); }
  XLValue getIfXLValue() const { return dyn_cast<XLValue>(getStorageL()); }
  DLValue getIfDLValue() const { return dyn_cast<DLValue>(getStorageL()); }

private:
  // These are named getStorageL instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageL() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageL() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// LValue = MLValue|XLValue|DLValue.
class LValue : public VariantValueStorage<LValue>,
               public VariantLValue<LValue> {
public:
  using VariantLValue::VariantLValue;
  using VariantValueStorage::VariantValueStorage;

  static LValue getFrom(Storage storage) {
    LValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<MLValue, XLValue, DLValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in a MLValue to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, LValue value);

template <typename DerivedType>
struct VariantBValue {
  VariantBValue() = default;
  VariantBValue(SBValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(MBValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(XBValue value) {
    if (value)
      getStorageB() = value;
  }
  VariantBValue(PValue value) {
    if (value)
      getStorageB() = value;
  }

  SBValue getIfSBValue() const { return dyn_cast<SBValue>(getStorageB()); }
  MBValue getIfMBValue() const { return dyn_cast<MBValue>(getStorageB()); }
  XBValue getIfXBValue() const { return dyn_cast<XBValue>(getStorageB()); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorageB()); }

private:
  // These are named getStorageB instead of getStorage to easy
  // multiple-inheritance name lookup issues.
  typename VariantValueStorage<DerivedType>::Storage &getStorageB() {
    return static_cast<DerivedType *>(this)->getStorage();
  }
  const typename VariantValueStorage<DerivedType>::Storage &
  getStorageB() const {
    return static_cast<const DerivedType *>(this)->getStorage();
  }
};

/// BValue = SBValue|MBValue|XBValue|PValue.
class BValue : public VariantValueStorage<BValue>,
               public VariantBValue<BValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantValueStorage::VariantValueStorage;

  static BValue getFrom(Storage storage) {
    BValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<SBValue, MBValue, XBValue, PValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in a MBValue to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, LValue value);

/// Concrete Value: CValue = CRValue|LValue|BValue.
class CValue : public VariantValueStorage<CValue>,
               public VariantCRValue<CValue>,
               public VariantLValue<CValue>,
               public VariantBValue<CValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantCRValue::VariantCRValue;
  using VariantLValue::VariantLValue;
  using VariantValueStorage::VariantValueStorage;

  CValue() = default;
  CValue(CRValue value) { getStorage() = value.getStorage(); }
  CValue(BValue value) { getStorage() = value.getStorage(); }
  CValue(LValue value) { getStorage() = value.getStorage(); }
  CValue(PValue value) {
    if (value)
      storage = value;
  }

  static CValue getFrom(Storage storage) {
    CValue result;
    // Initialize conditionally based on what is in Storage.
    if (isa<PValue, SRValue, MRValue, XRValue, SBValue, MBValue, XBValue,
            MLValue, XLValue, DLValue>(storage))
      result.storage = std::move(storage);
    return result;
  }

  BValue getIfBValue() const { return BValue::getFrom(getStorage()); }
  RValue getIfRValue() const { return RValue::getFrom(getStorage()); }
  LValue getIfLValue() const { return LValue::getFrom(getStorage()); }
  CRValue getIfCRValue() const { return CRValue::getFrom(storage); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorage()); }

  /// Return the type for the contained representation, or null if null.
  ASTType getType() const;

  /// This method looks through the pointer in memory references to return
  /// the underlying type.
  ASTType getRValueType() const;
  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, CRValue value);

/// AnyValue = LValue|BValue|RValue.
class AnyValue : public VariantValueStorage<AnyValue>,
                 public VariantCRValue<AnyValue>,
                 public VariantLValue<AnyValue>,
                 public VariantBValue<AnyValue>,
                 public VariantURValue<AnyValue> {
public:
  using VariantBValue::VariantBValue;
  using VariantCRValue::VariantCRValue;
  using VariantLValue::VariantLValue;
  using VariantURValue::VariantURValue;
  using VariantValueStorage::VariantValueStorage;

  AnyValue() = default;

  AnyValue(URValue value) { storage = value.getStorage(); }
  AnyValue(CRValue value) { storage = value.getStorage(); }
  AnyValue(BValue value) { storage = value.getStorage(); }
  AnyValue(RValue value) { storage = value.getStorage(); }
  AnyValue(LValue value) { storage = value.getStorage(); }
  AnyValue(CValue value) { storage = value.getStorage(); }
  AnyValue(PValue value) {
    if (value)
      storage = value;
  }

  LValue getIfLValue() const { return LValue::getFrom(storage); }
  URValue getIfURValue() const { return URValue::getFrom(storage); }
  CRValue getIfCRValue() const { return CRValue::getFrom(storage); }
  CValue getIfCValue() const { return CValue::getFrom(storage); }
  RValue getIfRValue() const { return RValue::getFrom(storage); }
  BValue getIfBValue() const { return BValue::getFrom(storage); }
  PValue getIfPValue() const { return dyn_cast<PValue>(getStorage()); }

  void dump() const;
};
raw_ostream &operator<<(raw_ostream &os, AnyValue value);

/// A shorthand to make function operand handling more readable.
using FuncOperand = ASTExprAnd<AnyValue>;

//===----------------------------------------------------------------------===//
// BaseDLValue classes.
//===----------------------------------------------------------------------===//

/// Subclasses of BaseDLValue model a dynamic LValue which has a computed getter
/// and setter.
class BaseDLValue : public NonAtomicallyReferenceCounted<BaseDLValue> {
public:
  /// This is the RValue type of the value being accessed if known.  It is
  /// inferred from the get/set.
  ASTType elementType;

  BaseDLValue(ASTType elementType) : elementType(elementType) {}

  virtual ~BaseDLValue();
  virtual void print(raw_ostream &os) const = 0;
  virtual CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const = 0;
  virtual void emitStore(ASTExprAnd<CValue> value,
                         ExprEmitter &emitter) const = 0;
};

/// This DLValue implementation represents a discard pattern of _.  It discards
/// its result on store and produces an error if attempting to load it.
class DiscardDLValue : public BaseDLValue {
public:
  const ExprNode *expr;

  DiscardDLValue(ASTType elementType, const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents a stored attribute projected from
/// another DLValue, e.g. `swap(&a[i].x, ...)`.
class StoredAttributeRefDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  ASTExprAnd<DLValue> baseVal;
  Operation *fieldOp; // StructFieldOp

  StoredAttributeRefDLValue(ASTExprAnd<DLValue> baseVal, StructFieldOp fieldOp,
                            ASTType elementType, const ExprNode *expr);

  StructFieldOp getField() const;

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents property access `a.x =`
/// and with subscript syntax `a[i,j] = `, invoking __getattr__/__setattr__ and
/// __getitem__ and __setitem__ respectively.
///
/// We allow DLValues to have getter+setter or just setter.
class SubscriptDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  // Positional operands (including self) for the setter/getter call.
  SmallVector<FuncOperand> posOperands;
  // Keyword operands for the setter/getter call.
  SmallDenseMap<StringAttr, FuncOperand> kwOperands;

  /// Return true if this is a subscript, false if this is an attribute access.
  bool isSubscript() const;

  SubscriptDLValue(SmallVectorImpl<FuncOperand> &&posOperands,
                   SmallDenseMap<StringAttr, FuncOperand> &&kwOperands,
                   ASTType elementType, const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents tuple lvalues, e.g. `(a[i], b) = x`.
class TupleDLValue : public BaseDLValue {
public:
  const ExprNode *expr;
  // These are the LValues for the sub-elements.
  std::vector<FuncOperand> eltLValues;

  TupleDLValue(ArrayRef<FuncOperand> eltLValues, ASTType tupleType,
               const ExprNode *expr);

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

/// This DLValue implementation represents a global variable reference.
class GlobalDLValue : public BaseDLValue {
public:
  /// The global variable operation.
  Operation *op;
  llvm::SMLoc loc;

  GlobalDLValue(GlobalVarDeclOp op, ASTType type, llvm::SMLoc loc);

  GlobalVarDeclOp getGlobal() const;

  void print(raw_ostream &os) const override;
  CValue emitLoad(ValueDest &dest, ExprEmitter &emitter) const override;
  void emitStore(ASTExprAnd<CValue> value, ExprEmitter &emitter) const override;
};

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
struct PointerLikeTypeTraits<M::KGEN::LIT::MLValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MLValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::XLValue>
    : public MLIRValueWrapper<M::KGEN::LIT::XLValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::SRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::SRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::XRValue>
    : public MLIRValueWrapper<M::KGEN::LIT::XRValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::SBValue>
    : public MLIRValueWrapper<M::KGEN::LIT::SBValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::MBValue>
    : public MLIRValueWrapper<M::KGEN::LIT::MBValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::XBValue>
    : public MLIRValueWrapper<M::KGEN::LIT::XBValue> {};

template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::PValue> {
public:
  using PValue = M::KGEN::LIT::PValue;
  static const void *getAsVoidPointer(PValue value) {
    return value.get().getAsOpaquePointer();
  }
  static PValue getFromVoidPointer(void *pointer) {
    return PValue(cast_or_null<mlir::TypedAttr>(
        mlir::Attribute::getFromOpaquePointer(pointer)));
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable
  };
};
} // namespace llvm

#endif // KGEN_MOJOPARSER_IRVALUES_H
