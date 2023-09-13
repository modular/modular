//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LOWERTOOBJECTIMPL_H
#define LOWERTOOBJECTIMPL_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {
class Module;
class TargetMachine;
} // namespace llvm

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN {
class CompilationOptions;

/// Compile the given LLVM module to an object file and write it to objStream.
LogicalResult compileLLVMToObject(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine,
                                  llvm::raw_pwrite_stream &objStream,
                                  CompilationOptions &options,
                                  LLCL::Runtime &runtime,
                                  bool emitAssembly = false);
} // namespace M::KGEN

#endif // LOWERTOOBJECTIMPL_H
