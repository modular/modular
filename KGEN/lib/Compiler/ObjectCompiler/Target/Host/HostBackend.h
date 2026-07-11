//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// TargetBackend for host (CPU) targets: llc to an object, then link a shared
// object. The default backend for host (x86 and AArch64) triples, and the
// emission base reused by GPU backends whose object lowering matches
// the host path.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H
#define KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H

#include "KGEN/Compiler/Target/TargetBackend.h"

namespace M::KGEN {

class HostBackend : public TargetBackend {
public:
  const TargetTraits *traits() const override;

  SplitStrategy
  splitStrategy(const CompilationOptions &options) const override {
    return options.enableLLVMPerFunctionSplitting ? SplitStrategy::PerFunction
                                                  : SplitStrategy::PerExported;
  }

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override;
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override;
  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override;
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H
