//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_HELPERS_H
#define KGEN_LIB_MOGGPREELAB_HELPERS_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN::MOGGPreElab {

namespace {

template <typename LambdaToApply>
SmallVector<TypedAttr> forEachDecorator(GeneratorOp userKernel,
                                        LambdaToApply lambda) {
  SmallVector<TypedAttr> decoratorsToCopy;
  for (TypedAttr decorator : userKernel.getDecorators()) {
    // Keep track of the non mogg decorators to preserve them on the user
    // kernel.
    decoratorsToCopy.push_back(decorator);

    // Decorators are expected to the the apply of a symbol.
    auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator);
    if (!apply)
      continue;

    // The first operand is expected to be the symbol we are applying.
    auto sym = dyn_cast<KGEN::SymbolConstantAttr>(apply.getOperand(0));
    if (!sym)
      continue;

    StringRef decoratorName = sym.getSymbol().getLeafReference().strref();
    lambda(decorator, decoratorName, decoratorsToCopy);
  }
  return decoratorsToCopy;
}

} // namespace

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H
