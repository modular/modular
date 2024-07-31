//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_MOGGUTILS_H
#define KGEN_LIB_MOGGPREELAB_MOGGUTILS_H

#include "mlir/IR/BuiltinAttributes.h"

#include <array>

namespace M::KGEN::MOGGPreElab {

/// Returns true if the symbol's references (files/modules it came from) match
/// the provided path.
template <unsigned long N>
inline bool symbolMatches(mlir::SymbolRefAttr symbol,
                          const std::array<llvm::StringLiteral, N> &path) {
  if (symbol.getNestedReferences().size() != N - 1)
    return false;

  if (!symbol.getRootReference().strref().contains(path.front()))
    return false;

  for (auto [i, ref] : llvm::enumerate(symbol.getNestedReferences())) {
    if (!ref.getValue().contains(path[i + 1]))
      return false;
  }

  return true;
}
} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_MOGGUTILS_H
