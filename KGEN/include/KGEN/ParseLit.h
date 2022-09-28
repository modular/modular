//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PARSELIT_H
#define KGEN_PARSELIT_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace llvm {
class SourceMgr;
}
namespace mlir {
class TimingScope;
}

namespace M {

/// Parse a single .lit file and return the MLIR module for it.
OwningOpRef<ModuleOp> importLitFile(llvm::SourceMgr &sourceMgr,
                                    MLIRContext *context,
                                    mlir::TimingScope &ts);

} // namespace M

#endif // KGEN_PARSELIT_H
