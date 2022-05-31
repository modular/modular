//===- GenericML/Support/TensorEltType.h ----------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorEltType which is a standardized type to
// represent the storage EltType of tensors.  This does not hold quantization
// information.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TENSORELTTYPE_H
#define GENERICML_SUPPORT_TENSORELTTYPE_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace M {
template <typename ResultType>
class TensorEltTypeSwitch;

/// This classes represents the storage type for values in a tensor.  This is
/// encoded in a specific way intended to allow efficient analysis and
/// transformation of a type.
///
/// Note that this class is designed to allow open extension by ML frameworks by
/// subclassing it and adding "mOther" types.  This acknowledges that frameworks
/// will want to do weird things within their type system, e.g. have string or
/// ragged tensors (which ...aren't tensors!), have "resources", etc.  Given
/// this reality, we don't want to have conversions back and forth between enum
/// types all the time.  Extensibility allows use of this generic type for those
/// framework defined cases as well.
class TensorEltType {
public:
  /// Note that many of these should be interpreted as type classes, i.e. the
  /// enum represents a mask, not a concrete enumerator.  These enum values
  /// are prefixed with `m` to designate this.
  enum Cases : uint8_t {
    invalid = 0,

    //------ Encoding for ordinary primitives --------------------------------//

    // Bit 7 encode primitive category: 0 = Float/Other, 1 = SInt/UInt
    mIsInteger = 1 << 7,

    // Bit 6 encode for Float/Other category encodes "isFloat".
    mIsFloat = 1 << 6,

    // Bit 5 for integer and floating point types indicate if type is complex.
    // This keeps the element types densely packed, allowing table lookups.
    // Note that we support many integer and floating point element types in
    // complex number, but they must be at least a byte in size:
    //   `complex si1` is not supported (but `complex kBool` is).
    mIsComplex = 1 << 5,

    //===--- Signed and Unsigned Integer Types ----------------------------===//
    // This supports any power-of-two integer type up to a larger width than
    // MLIR supports.  The width is encoded in logarithmic form, which enables
    // small lookup tables indexed by the enum value.

    /// Bit 0 encodes "isSigned".
    mIsSigned = 1,

    kIntWidthShift = 1,
    // i1's densely packed in memory.
    si1 = (0 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui1 = (0 << kIntWidthShift) | mIsInteger,
    si2 = (1 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui2 = (1 << kIntWidthShift) | mIsInteger,
    si4 = (2 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui4 = (2 << kIntWidthShift) | mIsInteger,
    si8 = (3 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui8 = (3 << kIntWidthShift) | mIsInteger,
    si16 = (4 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui16 = (4 << kIntWidthShift) | mIsInteger,
    si32 = (5 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui32 = (5 << kIntWidthShift) | mIsInteger,
    si64 = (6 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui64 = (6 << kIntWidthShift) | mIsInteger,
    si128 = (7 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui128 = (7 << kIntWidthShift) | mIsInteger,

    //===--- Floating point types -----------------------------------------===//

    /// Bits 2+ indicate the kind of FP value.  The initial values are IEEE
    /// floating point with their FPKind value encoding the width of the float
    /// in bytes (logarithmic).
    f8 = 0 | mIsFloat,
    f16 = 1 | mIsFloat,
    f32 = 2 | mIsFloat,
    f64 = 3 | mIsFloat,
    f128 = 4 | mIsFloat,

    bf16 = 5 | mIsFloat,
    f24 = 6 | mIsFloat,
    f80 = 7 | mIsFloat,
    tf32 = 8 | mIsFloat,

    //===--- Encodings for other types ------------------------------------===//

    // kBool != ui1.  Like it, this only contains 1-bit of data, but it occupies
    // 1-byte of storage.  The rest of the byte is guaranteed to be zeros.
    kBool = 1,

    kFirstExtendedOption = 2,

    // Derived enums may add their own types into the "Other" category.  This
    // should always be done by starting with `kFirstExtendedOption`, e.g.
    // like this:
    //    kYourThing  = kFirstExtendedOption,
    //    kYourOtherThing,
  };

  // Related constants.
  enum {
    // Complex i128/f128 is the largest size thing this type enum can represent.
    // Knowing this allows kernels to use fixed-size on-stack buffers.
    kMaxElementSizeInBits = 128 * 2,
    kMaxElementSizeInBytes = kMaxElementSizeInBits / 8
  };

  /*implicit*/ constexpr TensorEltType() : value(invalid) {}
  /*implicit*/ constexpr TensorEltType(Cases v) : value(v) {}
  explicit constexpr TensorEltType(uint8_t v) : value((Cases)v) {}

  /// This returns the underlying integer value for the enum.
  constexpr uint8_t getValue() const { return value; }

  // Categorization.
  constexpr bool isInvalid() const { return value == invalid; }
  bool isInt() const { return (value & mIsInteger) != 0; }
  bool isSInt() const { return isInt() & ((value & mIsSigned) != 0); }
  bool isUInt() const { return isInt() & ((value & mIsSigned) == 0); }
  bool isFloat() const { return !isInt() & ((value & mIsFloat) != 0); }
  bool isOther() const { return !isInt() & ((value & mIsFloat) == 0); }

  // Complex number handling.
  constexpr bool isComplex() const { return value & mIsComplex; }

  /// If the current type is a complex type, remove the complex marker to get
  /// the underlying element type, otherwise return it unmodified.
  constexpr TensorEltType stripComplex() const {
    return TensorEltType(value & ~mIsComplex);
  }

  /// Given a valid element type for a complex number, return the complex type.
  /// We do not support sub-byte element types in order to simplify clients.
  constexpr static TensorEltType getComplex(TensorEltType eltType) {
    assert(eltType.getWidthInBits() >= 8 && "invalid element type for complex");
    assert(!eltType.isComplex() &&
           "cannot construct a complex type with complex type as element");
    return TensorEltType(eltType.getValue() | mIsComplex);
  }

  // Integer handling.

  /// This returns a TensorEltType for an integer with the specified width and
  /// signedness, or `invalid` if it cannot be represented.
  static FailureOr<TensorEltType> getInt(unsigned widthInBits, bool isSigned) {
    if (!llvm::isPowerOf2_32(widthInBits) ||
        // Disallow large numbers by policy, because we don't want clients to
        // have to worry about memory allocation for arbitrarily large values.
        widthInBits > kMaxElementSizeInBits / 2)
      return failure();
    unsigned widthEncoding = llvm::countTrailingZeros(widthInBits);
    return TensorEltType((widthEncoding << kIntWidthShift) | mIsInteger |
                         (isSigned ? mIsSigned : 0));
  }

  /// Return the width in bits of this type, assuming it is an integer type.
  size_t getIntegerWidthInBits() const {
    return size_t(1) << getIntegerWidthInLogBits();
  }

  /// Return the width in bits of this type, assuming it is an integer type.
  /// This returns the amount in power of two, e.g. it returns:
  ///    0 == 1, 1 == 2, 2 == 4, 3 == 8, etc.
  size_t getIntegerWidthInLogBits() const {
    assert(isInt() && "Can only get the width of an integer type");
    return (value & ~(mIsInteger | mIsComplex)) >> kIntWidthShift;
  }

  // Type generic methods.

  /// Return the width of this element in bits.  This returns -1 for unknown
  /// width values.
  constexpr ssize_t getWidthInBits() const;

  /// Return the in-memory size for an array of the specified type with the
  /// specified number of elements, or -1 for non-numeric types or too large
  /// values.  This supports densely packed sub-byte types like i1, i2, i4.
  ssize_t getSizeInBytes(size_t numElements) const;

  constexpr bool operator==(uint8_t v) const { return value == v; }
  constexpr bool operator!=(uint8_t v) const { return value != v; }
  constexpr bool operator==(TensorEltType v) const { return value == v.value; }
  constexpr bool operator!=(TensorEltType v) const { return value != v.value; }

  /// Perform a eltType dispatch to delegate to a lambda or other callable, see
  /// the definition of `TensorEltTypeSwitch` below.
  template <typename ResultType>
  TensorEltTypeSwitch<ResultType> dispatch(void *bufferPtr) const;

  /// Return a string form of this eltType suitable for printing and error
  /// messages.
  std::string getAsString() const;

private:
  Cases value;
};

static_assert(sizeof(TensorEltType) == 1, "TensorEltType should not grow");

//===----------------------------------------------------------------------===//
// Method implementation for constexpr methods.
//===----------------------------------------------------------------------===//

/// Return the width of this element in bits.  This returns 0 for unknown
/// width values.
inline constexpr ssize_t TensorEltType::getWidthInBits() const {
  // Handle complex separately from per-element types below.  We know that
  // complex element types are always at least a byte in size.
  if (isComplex())
    return stripComplex().getWidthInBits() * 2;

  // This switch handles special cases inline, or determines the logrithmic size
  // of each element and breaks for the overflow check.
  switch (getValue()) {
  default:
    return isInt() ? getIntegerWidthInBits() : -1;
    // Handle other types.
  case TensorEltType::f8:
  case TensorEltType::kBool:
    return 8;
  case TensorEltType::f16:
  case TensorEltType::bf16:
    return 16;
  case TensorEltType::f32:
  case TensorEltType::tf32:
    return 32;
  case TensorEltType::f64:
    return 64;
  case TensorEltType::f80:
    return 80;
  }
}

//===----------------------------------------------------------------------===//
// TensorEltTypeForCXXType CXXTypeForTensorEltType
//===----------------------------------------------------------------------===//

/// Provide a mapping from C++ types to the corresponding EltType kinds.
template <typename CXXType>
struct TensorEltTypeForCXXType {
  // Default mapping allows a static_assert instead of template substitution
  // failure to catch invalid cases.
  static constexpr TensorEltType kind = TensorEltType::invalid;
};

/// Provide a mapping from TensorEltType enums to C++ types.
template <unsigned TensorEltType>
struct CXXTypeForTensorEltType {
  using CXXType = void;
};

#define DECLARE_TYPE_MAPPING(ELTTYPE_KIND, CXXTYPE)                            \
  template <>                                                                  \
  struct TensorEltTypeForCXXType<CXXTYPE> {                                    \
    static constexpr TensorEltType kind = TensorEltType::ELTTYPE_KIND;         \
  };                                                                           \
  template <>                                                                  \
  struct CXXTypeForTensorEltType<TensorEltType::ELTTYPE_KIND> {                \
    using CXXType = CXXTYPE;                                                   \
  };

DECLARE_TYPE_MAPPING(kBool, bool);
DECLARE_TYPE_MAPPING(si8, int8_t);
DECLARE_TYPE_MAPPING(ui8, uint8_t);
DECLARE_TYPE_MAPPING(si16, int16_t);
DECLARE_TYPE_MAPPING(ui16, uint16_t);
DECLARE_TYPE_MAPPING(si32, int32_t);
DECLARE_TYPE_MAPPING(ui32, uint32_t);
DECLARE_TYPE_MAPPING(si64, int64_t);
DECLARE_TYPE_MAPPING(ui64, uint64_t);
DECLARE_TYPE_MAPPING(f32, float);
DECLARE_TYPE_MAPPING(f64, double);
/// TODO: Add long double when sizeof(long double) != sizeof(double).
#undef DECLARE_TYPE_MAPPING

//===----------------------------------------------------------------------===//
// TensorEltTypeSwitch
//===----------------------------------------------------------------------===//

/// This class is used to implement switch-like dispatch for TensorEltType
/// values. In addition to allowing checks of enumerator values, it provides
/// convenients helpers for things like "C++ floating point types" and "integer
/// types ignoring sign" etc.  This should be used like:
///
///   someTensorEltType.dispatch<>(bufferPtr)  // pass in void*
///      .when<TensorEltType::f24>([](void *buf) { ... invoked when f24 ... })
///      .when([](bool *bufPtr) { ... invoked when kBool ... })
///      .whenCXXFP([](auto *bufPtr) { ... invoked with correct pointer type ...
///      .otherwise([]() { ... invoked otherwise ... });
///
/// TODO: Generalize this to taking more than one pointer, casting all of them
/// at the same time!
template <typename ResultType>
class TensorEltTypeSwitch {
public:
  TensorEltTypeSwitch(TensorEltType value, void *bufferPtr)
      : bufferPtr(bufferPtr), value(value) {}
  ~TensorEltTypeSwitch() = default;

  /// Add a case on the given type.
  template <uint8_t CaseValue, typename CallableT>
  TensorEltTypeSwitch &when(CallableT &&caseFn) {
    using CXXType = typename CXXTypeForTensorEltType<CaseValue>::CXXType;
    // Check to see if any of the types apply to 'value'.
    if (!result && this->value.getValue() == CaseValue)
      result = caseFn(static_cast<CXXType *>(bufferPtr));
    return *this;
  }

  /// Invoke a case on the derived class, inferring the type of the Case from
  /// the first input of the given callable.
  /// Note: This inference rules for this overload are very simple: strip
  ///       pointers and references.
  template <typename CallableT>
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG TensorEltTypeSwitch &
  when(CallableT &&caseFn) {
    using Traits = llvm::function_traits<std::decay_t<CallableT>>;
    using ElementType = std::remove_cv_t<std::remove_pointer_t<
        std::remove_reference_t<typename Traits::template arg_t<0>>>>;
    constexpr auto kind = TensorEltTypeForCXXType<ElementType>::kind.getValue();
    static_assert(kind != TensorEltType::invalid,
                  "unknown C++ pointer type in lambda");
    return this->when<kind>(std::forward<CallableT>(caseFn));
  }

  /// Invoke a case on the derived class with multiple case types.
  template <uint8_t CaseV1, uint8_t CaseV2, uint8_t... CaseVs,
            typename CallableT>
  // This is marked always_inline and nodebug so it doesn't show up in stack
  // traces at -O0 (or other optimization levels).  Large TensorEltTypeSwitch's
  // are common, are equivalent to a switch, and don't add any value to stack
  // traces.
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG TensorEltTypeSwitch &
  when(CallableT &&caseFn) {
    return when<CaseV1>(caseFn).template when<CaseV2, CaseVs...>(caseFn);
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  TensorEltTypeSwitch &otherwise(CallableT &&defaultFn) {
    if (!result)
      result = defaultFn();
    return *this;
  }

  /// Invoke the specified lambda with all the standard C++ integer types from 8
  /// to 64 bits in both signed and unsigned forms.  This passes the pointer in
  /// with the correct C++ type, so it is usually best to use a generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr)
  ///     .whenCXXInt([&](auto *ptr) { use ptr generically })
  ///
  template <typename CallableT>
  TensorEltTypeSwitch &whenCXXInt(CallableT &&elementFn) {
    return when<TensorEltType::si8, TensorEltType::ui8, TensorEltType::si16,
                TensorEltType::ui16, TensorEltType::si32, TensorEltType::ui32,
                TensorEltType::si64, TensorEltType::ui64>(
        std::forward<CallableT>(elementFn));
  }

  /// Invoke the specified lambda with `float` and `double`.  This passes the
  /// pointer in with the correct C++ type, so it is usually best to use a
  /// generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr)
  ///     .whenCXXInt([&](auto *ptr) { use ptr generically })
  ///
  /// TODO: Add long double when sizeof(long double) != sizeof(double).
  template <typename CallableT>
  TensorEltTypeSwitch &whenCXXFP(CallableT &&elementFn) {
    return when<TensorEltType::f32, TensorEltType::f64>(
        std::forward<CallableT>(elementFn));
  }

  LLVM_NODISCARD
  operator ResultType() {
    assert(result && "Fell off the end of a TensorEltTypeSwitch");
    return std::move(*result);
  }

private:
  /// TensorEltTypeSwitch is not a value.
  TensorEltTypeSwitch(const TensorEltTypeSwitch &) = delete;
  TensorEltTypeSwitch(TensorEltTypeSwitch &&other) = delete;
  void operator=(const TensorEltTypeSwitch &) = delete;
  void operator=(TensorEltTypeSwitch &&other) = delete;

  void *const bufferPtr;     /// The buffer pointer that we're casting.
  const TensorEltType value; /// The value we are switching on.

  /// The result of this switch statement, once known, None before that.
  llvm::Optional<ResultType> result;
};

/// Perform a eltType dispatch to delegate to a lambda or other callable, see
/// the definition of `TensorEltTypeSwitch` below.
template <typename ResultType>
inline TensorEltTypeSwitch<ResultType>
TensorEltType::dispatch(void *bufferPtr) const {
  return TensorEltTypeSwitch<ResultType>(*this, bufferPtr);
}

} // end namespace M

#endif // GENERICML_SUPPORT_TENSORELTTYPE_H
