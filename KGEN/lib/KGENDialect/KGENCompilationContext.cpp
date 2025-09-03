//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_COMPILATIONCONTEXT_CPP
#define KGEN_KGENDIALECT_COMPILATIONCONTEXT_CPP

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

#endif // KGEN_KGENDIALECT_COMPILATIONCONTEXT_CPP
