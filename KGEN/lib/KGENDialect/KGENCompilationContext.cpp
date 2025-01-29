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
    std::visit(
        [&](auto &&v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, bool>) {
            os << v;
          } else if constexpr (std::is_same_v<T, int>) {
            os << v;
          } else if constexpr (std::is_same_v<T, std::string>) {
            os << v;
          } else {
            // NOTE: This should be a static_assert, but that breaks in torch
            // compile tests on some mac devices.
            assert("non-exhaustive visitor!");
          }
          os << ';';
        },
        entry.second);
  }
}

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_COMPILATIONCONTEXT_CPP
