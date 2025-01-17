//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_COMPILATIONCONTEXT_H
#define KGEN_KGENDIALECT_COMPILATIONCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include <variant>

namespace M::KGEN {

/// Represents kernel compilation options that is used to set mojo parameters
/// during JITing.
struct CompilationContext {
  llvm::DenseMap<llvm::StringRef, std::variant<bool, int, llvm::StringRef>>
      mojoDefines;
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_COMPILATIONCONTEXT_H
