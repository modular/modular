//===- KGEN/InitAllDialects.h ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INITALLDIALECTS_H
#define KGEN_INITALLDIALECTS_H

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
// Add all the MLIR dialects to the provided registry.
inline void registerAllKGENDialects(DialectRegistry &registry) {
  registry.insert<KGEN::KGENDialect>();
  registry.insert<KGEN::MetaDialect>();
}

} // namespace M

#endif // KGEN_INITALLDIALECTS_H
