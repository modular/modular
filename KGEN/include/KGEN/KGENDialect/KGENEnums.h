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

//===----------------------------------------------------------------------===//
// ArgConvention
//===----------------------------------------------------------------------===//

/// Determine whether an argument with the given input convention expects to
/// have a pointer or reference type.
static inline bool hasAddress(ArgConvention conv) {
  return conv != ArgConvention::OwnedReg && conv != ArgConvention::ReadReg;
}

/// Determine whether an argument with the given input convention expects to
/// have an implicit origin.
static inline bool hasImplicitOrigin(ArgConvention conv) {
  switch (conv) {
  case ArgConvention::Ref:
  case ArgConvention::MutRef:
  case ArgConvention::OwnedReg:
  case ArgConvention::ReadReg:
    return false;
  case ArgConvention::OwnedMem:
  case ArgConvention::ReadMem:
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    return true;
  }
  llvm_unreachable("invalid argument convention");
}

/// Return true if this is an memory location for a normal or error result.
static inline bool isResultSlot(ArgConvention conv) {
  return conv == ArgConvention::ByRefResult ||
         conv == ArgConvention::ByRefError;
}

//===----------------------------------------------------------------------===//
// FnEffects
//===----------------------------------------------------------------------===//

/// This class represents the effects of a callable. A callable can throw an
/// error, be an async function, etc. The effect of a callable is load-bearing
/// on its type.
class FnEffects {
  using Impl = impl::FnEffects;

public:
  FnEffects(Impl impl = Impl::None) : impl(impl) {}

  bool isThrows() const { return get(Impl::Throws); }
  bool isAsync() const { return get(Impl::Async); }
  bool isCapturing() const { return get(Impl::Capturing); }
  bool isEscaping() const { return get(Impl::Escaping); }
  bool isRefResult() const { return get(Impl::RefResult); }
  bool isUnified() const { return get(Impl::Unified); }

  FnEffects setThrows(bool throws = true) { return set(Impl::Throws, throws); }
  FnEffects setAsync(bool async = true) { return set(Impl::Async, async); }
  FnEffects setCapturing(bool capturing = true) {
    return set(Impl::Capturing, capturing);
  }
  FnEffects setEscaping(bool escaping = true) {
    return set(Impl::Escaping, escaping);
  }
  FnEffects setRefResult(bool value = true) {
    return set(Impl::RefResult, value);
  }
  FnEffects setUnified(bool value = true) { return set(Impl::Unified, value); }

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
// ArgConvention
//===----------------------------------------------------------------------===//

template <typename StreamT>
inline StreamT &operator<<(StreamT &os, ArgConvention convention) {
  os << stringifyArgConvention(convention);
  return os;
}

/// Return a string like "read" or "mut".
const char *getUserSyntax(ArgConvention convention);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENENUMS_H
