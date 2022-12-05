//===----------------------------------------------------------------------===//
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
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"

namespace M {
// Add all the MLIR dialects to the provided registry.
inline void registerAllKGENDialects(DialectRegistry &registry) {
  registry.insert<KGEN::KGENDialect>();
  registry.insert<KGEN::LIT::LITDialect>();
  registry.insert<KGEN::POP::POPDialect>();
  registry.insert<KGEN::ZAP::ZAPDialect>();
  registry.insert<MDialect>();
}

} // namespace M

#endif // KGEN_INITALLDIALECTS_H
