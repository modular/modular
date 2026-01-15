//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENCompilationContext.h"

namespace M::KGEN {

void CompilationContext::print(llvm::raw_ostream &os) const {
  for (auto entry : mojoDefines) {
    auto k = entry.first;
    os << k << '=';
    std::visit([&](auto &&v) { os << v << ';'; }, entry.second);
  }
}

} // namespace M::KGEN
