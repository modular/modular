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
class FnMetadataAttr;
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

  VariadicEffects setKWVarArgs(bool kwVarArgs = true) {
    return set(Impl::KWVarArg, kwVarArgs);
  }

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

private:
  VariadicEffects set(Impl bit, bool value) {
    impl = VariadicImpl::bitEnumSet(impl, bit, value);
    return *this;
  }
  bool get(Impl bit) const {
    return VariadicImpl::bitEnumContainsAny(impl, bit);
  }

  Impl impl;

  friend FnMetadataAttr;
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
