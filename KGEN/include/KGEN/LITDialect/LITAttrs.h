//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Regex.h"

namespace M::KGEN {
class DeclRefType;
class NoneType;
namespace LIT {
class LifetimeType;
class MetaTypeType;
class StructFieldOp;
class UnpackedType;
} // namespace LIT
} // namespace M::KGEN

#include "KGEN/LITDialect/LITEnums.h.inc"

namespace M::KGEN::LIT {

/// This class represents the effects of a callable. A callable can throw an
/// error, be an async function, have different kinds of varargs, etc. The
/// effect of a callable is load-bearing on its type.
class VariadicEffects {
  using Impl = VariadicImpl::VariadicEffects;

public:
  VariadicEffects(Impl impl = Impl::None) : impl(impl) {}

  VariadicEffects setVarArgs(bool varArgs = true) {
    return set(Impl::VarArg, varArgs);
  }
  VariadicEffects setPackVarArgs(bool packVarArgs = true) {
    return set(Impl::PackVarArg, packVarArgs);
  }
  bool hasVarArgs() const { return get(Impl::VarArg); }
  bool hasPackVarArgs() const { return get(Impl::PackVarArg); }
  bool hasAnyVarArgs() const { return hasVarArgs() || hasPackVarArgs(); }

  VariadicEffects setKWVarArgs(bool kwVarArgs = true) {
    return set(Impl::KWVarArg, kwVarArgs);
  }
  bool hasKWVarArgs() const { return get(Impl::KWVarArg); }

  VariadicEffects setParamVarArgs(bool paramVarArgs = true) {
    return set(Impl::ParamVarArg, paramVarArgs);
  }
  bool hasParamVarArgs() const { return get(Impl::ParamVarArg); }

  bool operator==(VariadicEffects rhs) const {
    return getImpl() == rhs.getImpl();
  }
  bool operator!=(VariadicEffects rhs) const {
    return getImpl() != rhs.getImpl();
  }

  Impl getImpl() const { return impl; }

  /// Given a function with `numInputs` inputs, return true if the argument at
  /// `index` is the variadic argument.
  bool isVarArg(size_t numInputs, size_t index) {
    // If the function has keyword varargs, the vararg index is the second last.
    // Otherwise, it's the last.
    return (index + 1 + hasKWVarArgs()) == numInputs;
  }

private:
  VariadicEffects set(Impl bit, bool value) {
    impl = VariadicImpl::bitEnumSet(impl, bit, value);
    return *this;
  }
  bool get(Impl bit) const {
    return VariadicImpl::bitEnumContainsAny(impl, bit);
  }

  Impl impl;
};

template <typename StreamT>
inline StreamT &operator<<(StreamT &os, VariadicEffects effects) {
  os << VariadicImpl::stringifyVariadicEffects(effects.getImpl());
  return os;
}

namespace VariadicImpl {
inline VariadicEffects operator|=(VariadicEffects &lhs, VariadicEffects rhs) {
  return lhs = lhs | rhs;
}
} // namespace VariadicImpl

} // namespace M::KGEN::LIT

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

#endif // KGEN_LITDIALECT_LITATTRS_H
