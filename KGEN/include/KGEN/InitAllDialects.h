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

#include "KGEN/HLKGENDialect/HLKGENDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
// Add all the MLIR dialects to the provided registry.
inline void registerAllKGENDialects(DialectRegistry &registry) {
  registry.insert<KGEN::KGENDialect>();
  registry.insert<KGEN::HLKGENDialect>();
  registry.insert<KGEN::MetaDialect>();
  registry.insert<KGEN::POPDialect>();
  registry.insert<KGEN::ZAPDialect>();
}

} // namespace M

#endif // KGEN_INITALLDIALECTS_H
