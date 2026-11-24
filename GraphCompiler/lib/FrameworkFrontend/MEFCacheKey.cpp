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
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/Passes.h"
#include "GraphCompiler/Version.h"
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

/// Replaces the device-info mapping with the one bit of it that changes
/// codegen, so a digest describes the program rather than the machines that
/// were present when it was built. A split's two halves do not hold the same
/// devices, so the mapping's keys and values differ between them.
void canonicalizeDeviceMapping(mlir::ModuleOp module) {
  auto *ctx = module.getContext();
  auto dict =
      module->getAttrOfType<DictionaryAttr>(MO::DeviceInfoMapping::kAttrName);
  if (!dict)
    return;
  bool fused = false;
  for (NamedAttribute entry : dict) {
    if (auto info = dyn_cast<DeviceInfoAttr>(entry.getValue()))
      fused = fused || info.getTileBasedFusion();
  }
  module->setAttr(
      MO::DeviceInfoMapping::kAttrName,
      DictionaryAttr::get(ctx, {{StringAttr::get(ctx, "tile_based_fusion"),
                                 BoolAttr::get(ctx, fused)}}));
}

/// Drops the kernel-library paths a graph records. They point into the tree
/// that built the graph, and a split builds its artifact in one tree and
/// initializes it from another. What the paths hold is part of the identity,
/// and is hashed from the libraries a compile is handed instead (see
/// `feedKernelLibraries`).
///
/// The module's own body is enough: `max/python/max/graph/graph.py` appends
/// every graph op, subgraphs included, to the end of the module body, so none
/// are nested. Walking would descend through the graph bodies, which is all of
/// the module's size.
void stripKernelLibraryPaths(mlir::ModuleOp module) {
  for (MO::GraphOp graph : module.getOps<MO::GraphOp>())
    graph->removeAttr("_kernel_library_paths");
}

/// Drops the kernel declarations a graph carries, which `seed_kernel_decls`
/// copied from its kernel library (`MOGG::copyKernelDecls`). They are that
/// library's signatures, and the store digests the library's contents; a
/// declaration's platform predicate names the devices that were present when
/// the compile-only half ran, which the executing half does not share.
void stripKernelDecls(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> decls;
  for (auto decl : module.getOps<MOGG::KernelDeclOp>())
    decls.push_back(decl.getOperation());
  for (auto decl : module.getOps<MOGG::PrimitiveDeclOp>())
    decls.push_back(decl.getOperation());
  for (auto decl : module.getOps<MOGG::ShapeFunctionDeclOp>())
    decls.push_back(decl.getOperation());
  for (mlir::Operation *decl : decls)
    decl->erase();
}

/// Feeds a module's body into `hasher`, with locations stripped and names
/// normalized, so a key describes the program rather than where it was built.
///
/// `canonicalizeDevices` drops the device-info mapping's per-device detail,
/// keeping only what it says about codegen. A key for the local cache does not
/// want that: it names the devices it was built for.
ErrorOrSuccess feedModuleBody(mlir::ModuleOp module, llvm::BLAKE3 &hasher,
                              bool canonicalizeDevices = false) {
  auto stripDebugInfo = mlir::PassManager::on<mlir::ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  stripDebugInfo.addPass(mlir::createStripDebugInfoPass());
  mlir::OwningOpRef<mlir::ModuleOp> moduleOpNoDebug = module.clone();
  if (failed(stripDebugInfo.run(*moduleOpNoDebug)))
    return Error::getStaticString(
        "failed to strip location information from the module op");
  MOGG::normalizeAllNames(*moduleOpNoDebug);
  // A graph's `counter` numbers the symbolic dimension names it mints, so it
  // advances as work is done to the graph (a compile mints more) rather than
  // with what the graph computes. Pin it, or a second compile of one graph is
  // a cache miss.
  moduleOpNoDebug->walk([](MO::GraphOp graph) { graph.setCounter(0); });
  if (canonicalizeDevices) {
    stripKernelLibraryPaths(*moduleOpNoDebug);
    stripKernelDecls(*moduleOpNoDebug);
    canonicalizeDeviceMapping(*moduleOpNoDebug);
  }

  WriteableBufferRef moduleBuf = WriteableBuffer::get();
  if (failed(Cache::writeOperationToCacheKey(*moduleOpNoDebug, moduleBuf)))
    return Error::getStaticString("failed to write module op to cache key");

  hasher.update(
      llvm::StringRef(moduleBuf->getBufferStart(), moduleBuf->getBufferSize()));
  return success();
}

/// Feeds the contents of every kernel library in `paths` into `hasher`, in a
/// sorted order so the caller's order cannot matter.
ErrorOrSuccess feedKernelLibraries(llvm::BLAKE3 &hasher,
                                   llvm::ArrayRef<std::string> paths) {
  llvm::SmallVector<std::string> sorted(paths.begin(), paths.end());
  llvm::sort(sorted);
  for (const std::string &libraryPath : sorted) {
    if (auto err = feedFile(
            hasher, libraryPath,
            "failed to open file specified as additional kernel library: "))
      return err.takeError();
  }
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
  if (auto err = feedKernelLibraries(b3, gcPipelineOptions.kernelLibraryPaths);
      err.isError())
    return err.takeError();

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
  llvm::BLAKE3 finalizer = prefixHasher;
  if (auto err = feedModuleBody(module, finalizer); err.isError())
    return err.takeError();
  return finalizeHasher(finalizer);
}

ErrorOr<std::string>
computeMefContentDigest(mlir::ModuleOp module,
                        const MO::GCPipelineOptions &gcPipelineOptions,
                        ContextRef maxContext) {
  // What a program *is*: its body, the codegen options, and the build that
  // compiled it. Nothing about the machine that ran the compile, which is what
  // makes this comparable across one split's two halves.
  //
  // The kernel libraries are the store's to cover, not this function's: what a
  // caller names is a path, and a path is the machine. The store digs the
  // contents up where it can tell a source package from a built one, and
  // appends them to this digest.
  llvm::BLAKE3 hasher;
  BLAKE3Stream os(hasher);
  os << "mef-content-v1";
  // Deliberately not the compiler version string: it hashes the build's own
  // pieces, and the two halves of a split compile with two builds (a record
  // runs in the exec configuration, a replay in the target one). Which MAX
  // release produced an artifact is the store's to record, from the release
  // version, which those two agree on.
  // Deliberately not the KGEN compilation context: its Mojo defines include
  // the ones a compile-only run sets for the device it is not holding, and the
  // two halves of a split do not set the same ones. What those defines change
  // is the code inside the kernels, which their contents cover.
  if (auto err = feedModuleBody(module, hasher, /*canonicalizeDevices=*/true);
      err.isError())
    return err.takeError();
  return finalizeHasher(hasher);
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
