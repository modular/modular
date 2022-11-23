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

namespace M::KGEN {
struct CompilationOptions;

/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(TargetInfoAttr targetInfo,
                    const CompilationOptions &options, bool isJIT);

/// Compile the given LLVM module to an object file and write it to objStream.
LogicalResult compileLLVMToObject(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine,
                                  llvm::raw_pwrite_stream &objStream);
} // namespace M::KGEN

#endif // LOWERTOOBJECTIMPL_H
