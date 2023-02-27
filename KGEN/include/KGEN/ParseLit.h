//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PARSELIT_H
#define KGEN_PARSELIT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include <string>

namespace llvm {
class SourceMgr;
} // namespace llvm
namespace mlir {
class TimingScope;
} // namespace mlir

namespace M {
namespace KGEN {
class CompilationOptions;
} // namespace KGEN

/// Parse a single .lit file and return the MLIR module for it.
///
/// When `useMLIRDiagnostics` is true, this prints diagnostics through MLIR (so
/// MLIR features like -verify-diagnostics may be used).  When false, this
/// prints them through SourceMgr to get ranges and fixit hints.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
OwningOpRef<ModuleOp>
importLitFile(llvm::SourceMgr &sourceMgr, MLIRContext *context,
              mlir::TimingScope &ts, const KGEN::CompilationOptions &options,
              bool useMLIRDiagnostics,
              SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Parse a single .lit file and produce an appropriate document detailing the
/// API within the module. The generated documentation is piped into the
/// provided output stream, in markdown format.
LogicalResult generateLitDoc(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                             raw_ostream &outputOS, mlir::TimingScope &ts,
                             const KGEN::CompilationOptions &options);
} // namespace M

#endif // KGEN_PARSELIT_H
