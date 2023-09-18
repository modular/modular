//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the core KGEN attribute classes, provides implementation
// logic for working with them, and helpers for defining operations that take
// them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_H
#define KGEN_KGENDIALECT_KGENATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "Support/ErrorOr.h"
#include "Support/IPInt.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class OperationName;
} // namespace mlir

namespace M {
class BuildInfoAttr;
class TargetInfoAttr;

namespace KGEN {
class BuildInfoType;
class KGENDType;
class SignatureType;
class TargetType;
class VariadicType;
class VariadicAttr;

//===----------------------------------------------------------------------===//
// TypeConstantAttr
//===----------------------------------------------------------------------===//

/// Base class for MLIR type constant attributes. This attribute represents a
/// constant MLIR type expression.
class TypeConstantAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Returns the constant type value.
  Type getValue() const;

  /// Get a type constant attribute.
  static TypedAttr get(Type value);

  /// Returns true if the given type is classified as a concrete type.
  static bool isConcreteType(Type type);

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Enum Declarations
//===----------------------------------------------------------------------===//

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

//===----------------------------------------------------------------------===//
// FnEffects
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// This class represents the effects of a callable. A callable can throw an
/// error, be an async function, have different kinds of varargs, etc. The
/// effect of a callable is load-bearing on its type.
class FnEffects {
  using Impl = impl::FnEffects;

public:
  FnEffects(Impl impl = Impl::None) : impl(impl) {}

  FnEffects setThrows(bool throws = true) { return set(Impl::Throws, throws); }
  bool isThrows() const { return get(Impl::Throws); }

  FnEffects setAsync(bool async = true) { return set(Impl::Async, async); }
  bool isAsync() const { return get(Impl::Async); }

  FnEffects setVarArgs(bool varArgs = true) {
    return set(Impl::VarArg, varArgs);
  }
  FnEffects setPackVarArgs(bool packVarArgs = true) {
    return set(Impl::PackVarArg, packVarArgs);
  }
  bool hasVarArgs() const { return get(Impl::VarArg); }
  bool hasPackVarArgs() const { return get(Impl::PackVarArg); }
  bool hasAnyVarArgs() const { return hasVarArgs() || hasPackVarArgs(); }

  FnEffects setKWVarArgs(bool kwVarArgs = true) {
    return set(Impl::KWVarArg, kwVarArgs);
  }
  bool hasKWVarArgs() const { return get(Impl::KWVarArg); }

  FnEffects setParamVarArgs(bool paramVarArgs = true) {
    return set(Impl::ParamVarArg, paramVarArgs);
  }
  bool hasParamVarArgs() const { return get(Impl::ParamVarArg); }

  FnEffects setOwnedRegisterResult(bool ownedRegisterResult = true) {
    return set(Impl::OwnedResult, ownedRegisterResult);
  }
  bool hasOwnedRegisterResult() const { return get(Impl::OwnedResult); }

  FnEffects setCapturing(bool capturing = true) {
    return set(Impl::Capturing, capturing);
  }
  FnEffects setEscaping(bool escaping = true) {
    return set(Impl::Escaping, escaping);
  }
  bool isCapturing() const { return get(Impl::Capturing); }
  bool isEscaping() const { return get(Impl::Escaping); }

  bool operator==(FnEffects rhs) const { return getImpl() == rhs.getImpl(); }
  bool operator!=(FnEffects rhs) const { return getImpl() != rhs.getImpl(); }

  Impl getImpl() const { return impl; }

private:
  FnEffects set(Impl bit, bool value) {
    impl = impl::bitEnumSet(impl, bit, value);
    return *this;
  }
  bool get(Impl bit) const { return impl::bitEnumContainsAny(impl, bit); }

  Impl impl;
};
template <typename StreamT>
inline StreamT &operator<<(StreamT &os, FnEffects effects) {
  os << impl::stringifyFnEffects(effects.getImpl());
  return os;
}
namespace impl {
inline FnEffects operator|=(FnEffects &lhs, FnEffects rhs) {
  return lhs = lhs | rhs;
}
} // namespace impl

//===----------------------------------------------------------------------===//
// ValueInputConvention
//===----------------------------------------------------------------------===//

template <typename StreamT>
inline StreamT &operator<<(StreamT &os, ValueInputConvention convention) {
  os << stringifyValueInputConvention(convention);
  return os;
}

} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::ParamDeclRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamDeclRefAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamDeclRefAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<M::KGEN::ParamBindAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamBindAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamBindAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Emit an MLIR operation call in a parameter context.
TypedAttr emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
