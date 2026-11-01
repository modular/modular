//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MEFCacheKey.h"

#include "Cache/CachedTransform.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/Passes.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace M {

namespace {

class BLAKE3Stream : public llvm::raw_ostream {
public:
  explicit BLAKE3Stream(llvm::BLAKE3 &hasher) : hasher(hasher) {
    SetUnbuffered();
  }

private:
  void write_impl(const char *ptr, size_t size) override {
    hasher.update(llvm::StringRef(ptr, size));
    pos += size;
  }
  uint64_t current_pos() const override { return pos; }

  llvm::BLAKE3 &hasher;
  uint64_t pos = 0;
};

// Clears absolute-path options so the cache key is relocatable (MOCO-4152):
//   - kernel-library-paths: contents are hashed separately in `build`.
//   - save-temp-prefix / mlir-timing-out-dir: diagnostic outputs only.
//   - cache-dir: only set-vs-unset affects the artifact (CAS outlining), so
//     `build` folds in a "CAS enabled" bit instead.
void stripInstallPaths(MO::GCPipelineOptions &options) {
  options.cacheDir = "";
  options.kernelLibraryPaths = {};
  options.saveTempPrefix = "";
  options.mlirTimingOutDir = "";
}

void feedMOToMGPPipelineKey(llvm::BLAKE3 &hasher, mlir::MLIRContext *ctx,
                            const MO::GCPipelineOptions &options) {
  auto gcPipelinePasses = mlir::PassManager::on<mlir::ModuleOp>(
      ctx, mlir::OpPassManager::Nesting::Implicit);
  MO::buildGCPipeline(gcPipelinePasses, options);
  BLAKE3Stream os(hasher);
  gcPipelinePasses.printAsTextualPipeline(os);
}

ErrorOrSuccess feedFile(llvm::BLAKE3 &hasher, llvm::StringRef libraryPath,
                        llvm::StringRef errorPrefix) {
  auto bufferOr = M::openInputFile(libraryPath);
  if (bufferOr.isError())
    return Error(Twine(errorPrefix) + libraryPath);
  llvm::StringRef bytes = (*bufferOr)->getBuffer();
  hasher.update(bytes);
  return success();
}

std::string finalizeHasher(const llvm::BLAKE3 &hasher) {
  // `llvm_blake3_hasher_finalize` does not mutate the hasher.
  auto digest = const_cast<llvm::BLAKE3 &>(hasher).final();
  return std::string(reinterpret_cast<const char *>(digest.data()),
                     digest.size());
}

} // namespace

ErrorOr<MEFCacheKeyHasher>
MEFCacheKeyHasher::build(mlir::MLIRContext &mlirCtx,
                         const MO::GCPipelineOptions &gcPipelineOptions,
                         ContextRef maxContext) {
  MEFCacheKeyHasher hasher;
  llvm::BLAKE3 &b3 = hasher.prefixHasher;
  BLAKE3Stream os(b3);

  // Pipeline options + textual pipeline, path-stripped, plus a "CAS enabled"
  // bit. `GCPipelineOptions` is non-copyable, hence `copyOptionValuesFrom`.
  MO::GCPipelineOptions keyOptions;
  keyOptions.copyOptionValuesFrom(gcPipelineOptions);
  stripInstallPaths(keyOptions);
  keyOptions.print(os);
  os << "cas-enabled="
     << (gcPipelineOptions.cacheDir.hasValue() &&
         !gcPipelineOptions.cacheDir.getValue().empty());
  feedMOToMGPPipelineKey(b3, &mlirCtx, keyOptions);

  // KGEN compilation context (Mojo compile-time defines)
  if (auto *compilationCtx = maxContext->get<KGEN::CompilationContext>())
    compilationCtx->print(os);

  // Graph compiler version, pipeline, config.
  auto hostTargetOr = MOGG::getHostTargetInfoAttr(&mlirCtx);
  if (hostTargetOr.isError())
    return hostTargetOr.takeError();
  if (auto err = M::MOGG::graphCompilerCacheKey(os, hostTargetOr.takeValue()))
    return err.takeError();

  // User-provided kernel library files (ref MOCO-1570)
  for (const std::string &libraryPath : gcPipelineOptions.kernelLibraryPaths) {
    if (auto err = feedFile(
            b3, libraryPath,
            "failed to open file specified as additional kernel library: "))
      return err.takeError();
  }

  // Built-in kernel packages plus every binary package on the module import
  // path (see `getKernelPackagePathsForCacheKey`). Contents only, not paths
  // (MOCO-4152).
  auto kernelPackagePathsOr = MOGG::getKernelPackagePathsForCacheKey(&mlirCtx);
  if (kernelPackagePathsOr.isError())
    return kernelPackagePathsOr.takeError();
  for (const std::string &libraryPath : *kernelPackagePathsOr) {
    if (auto err = feedFile(b3, libraryPath, "failed to open kernel package: "))
      return err.takeError();
  }

  // Interpret config read failure as no config.
  auto configOr = M::Config::open();
  M::Config config = configOr.isError() ? M::Config() : configOr.takeValue();
  b3.update(
      llvm::formatv("max-debug.source-tracebacks={}",
                    config.getValueAsBool("max-debug.source-tracebacks", false))
          .str());

  return hasher;
}

ErrorOr<std::string>
MEFCacheKeyHasher::keyForModule(mlir::ModuleOp module) const {
  // Strip debug info and normalize symbols before hashing the IR
  auto stripDebugInfo = mlir::PassManager::on<mlir::ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  stripDebugInfo.addPass(mlir::createStripDebugInfoPass());
  mlir::OwningOpRef<mlir::ModuleOp> moduleOpNoDebug = module.clone();
  if (failed(stripDebugInfo.run(*moduleOpNoDebug)))
    return Error::getStaticString(
        "failed to strip location information from the module op");
  MOGG::normalizeAllNames(*moduleOpNoDebug);

  WriteableBufferRef moduleBuf = WriteableBuffer::get();
  if (failed(Cache::writeOperationToCacheKey(*moduleOpNoDebug, moduleBuf)))
    return Error::getStaticString("failed to write module op to cache key");

  llvm::BLAKE3 finalizer = prefixHasher;
  finalizer.update(
      llvm::StringRef(moduleBuf->getBufferStart(), moduleBuf->getBufferSize()));
  return finalizeHasher(finalizer);
}

ErrorOr<std::string>
computeMEFCacheKey(mlir::ModuleOp module,
                   const MO::GCPipelineOptions &gcPipelineOptions,
                   ContextRef maxContext) {
  auto hasherOr = MEFCacheKeyHasher::build(*module.getContext(),
                                           gcPipelineOptions, maxContext);
  if (hasherOr.isError())
    return hasherOr.takeError();
  return hasherOr.takeValue().keyForModule(module);
}

} // namespace M
