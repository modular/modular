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
namespace LLCL {
class Runtime;
} // namespace LLCL

/// This class provides the various configurations used to parse a .mojo file.
struct MojoParserConfig {
  MojoParserConfig(MLIRContext *context, LLCL::Runtime &runtime,
                   const KGEN::CompilationOptions &options)
      : context(context), runtime(runtime), options(options) {}

  /// The MLIR context to use when parsing the file.
  MLIRContext *context;

  /// The runtime to use when parsing the file.
  LLCL::Runtime &runtime;

  /// The compilation options to use when parsing the file.
  const KGEN::CompilationOptions &options;

  /// When true, this prints diagnostics through MLIR (so MLIR features like
  /// -verify-diagnostics may be used). When false, this prints them through
  /// SourceMgr to get ranges and fixit hints.
  bool useMLIRDiagnostics = false;

  /// If true, this will process and validate the doc strings in the file.
  bool validateDocStrings = false;
};

/// Parse a single .mojo file and return the MLIR module for it.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
OwningOpRef<ModuleOp>
importMojoFile(llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
               mlir::TimingScope &ts,
               SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Parse a single .mojo file and produce an appropriate document detailing the
/// API within the module. The generated documentation is piped into the
/// provided output stream, in markdown format.
LogicalResult generateMojoDoc(llvm::SourceMgr &sourceMgr,
                              MojoParserConfig &config, raw_ostream &outputOS,
                              mlir::TimingScope &ts);
} // namespace M

#endif // KGEN_PARSELIT_H
