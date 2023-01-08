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
namespace KGEN {
class CompilationOptions;
} // namespace KGEN

/// Parse a single .lit file and return the MLIR module for it.
///
/// When `useMLIRDiagnostics` is true, this prints diagnostics through MLIR (so
/// MLIR features like -verify-diagnostics may be used).  When false, this
/// prints them through SourceMgr to get ranges and fixit hints.
OwningOpRef<ModuleOp> importLitFile(llvm::SourceMgr &sourceMgr,
                                    MLIRContext *context, mlir::TimingScope &ts,
                                    const KGEN::CompilationOptions &options,
                                    bool useMLIRDiagnostics);
} // namespace M

#endif // KGEN_PARSELIT_H
