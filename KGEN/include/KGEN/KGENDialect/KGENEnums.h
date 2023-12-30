//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENENUMS_H
#define KGEN_KGENDIALECT_KGENENUMS_H

#include "mlir/IR/BuiltinAttributes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Enum Declarations
//===----------------------------------------------------------------------===//

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

namespace M::KGEN {

/// Enumeration of the compile emission format.
enum class EmissionKind : uint8_t { ASM, LLVM };

/// Return true if the specified input convention is passed with an implicit
/// lifetime.
inline bool isArgumentPassedWithImplicitLifetime(ValueInputConvention conv) {
  if (conv == ValueInputConvention::ByRefResult ||
      conv == ValueInputConvention::InitSelf ||
      conv == ValueInputConvention::OwnedInMem)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// FnEffects
//===----------------------------------------------------------------------===//

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

  /// Given a function with `numInputs` inputs, return true if the argument at
  /// `index` is the variadic argument.
  bool isVarArg(size_t numInputs, size_t index) {
    // If the function has keyword varargs, the vararg index is the second last.
    // Otherwise, it's the last.
    return (index + 1 + hasKWVarArgs()) == numInputs;
  }

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

#endif // KGEN_KGENDIALECT_KGENENUMS_H
