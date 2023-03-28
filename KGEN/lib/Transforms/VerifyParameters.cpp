//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENParameters.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Mutex.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_VERIFYPARAMETERS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct VerifyParametersPass : impl::VerifyParametersBase<VerifyParametersPass> {
  using VerifyParametersBase::VerifyParametersBase;

  void runOnOperation() override {
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    mlir::LockedSymbolTableCollection sharedSymtabs(analysis.getSymbolTables());
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

    // Give each thread a copy of the parameter cache, rather than each work
    // item.
    DenseMap<uint64_t, ParameterCollector::Analysis> threadCaches;
    threadCaches.reserve(getContext().getThreadPool().getThreadCount());
    llvm::sys::SmartRWMutex<true> mutex;

    std::vector<Region *> declRegions;
    for (auto decl : getOperation().getOps<DeclInterface>())
      for (Region &region : decl->getRegions())
        declRegions.push_back(&region);
    auto workFunc = [&sharedSymtabs, &paramCache, &mutex,
                     &threadCaches](Region *declRegion) {
      // Get the thread-local cache.
      ParameterCollector::Analysis *cache = nullptr;
      uint64_t threadId = llvm::get_threadid();
      {
        llvm::sys::SmartScopedReader<true> lock(mutex);
        auto it = threadCaches.find(threadId);
        if (it != threadCaches.end())
          cache = &it->second;
      }
      if (!cache) {
        llvm::sys::SmartScopedWriter<true> lock(mutex);
        // Each thread gets a copy of the saved cache.
        cache = &threadCaches.try_emplace(threadId, paramCache).first->second;
      }

      ParameterUseDefGraph graph(*declRegion);
      return graph.verify(sharedSymtabs, *cache);
    };
    if (failed(mlir::failableParallelForEach(&getContext(), declRegions,
                                             workFunc)))
      return signalPassFailure();

    // Consolidate the caches, but only when the original cache is empty. In
    // reality, the cache does not grow much after the first run of this pass on
    // an input IR, so consolidation is only worthwhile on the first run of the
    // pass, when the cache is empty.
    if (paramCache.parameterLess.empty()) {
      for (auto &[_, threadCache] : threadCaches)
        paramCache.parameterLess.insert(threadCache.parameterLess.begin(),
                                        threadCache.parameterLess.end());
    }

    // This pass does not modify any IR, so mark all analyses as preserved. In
    // addition, this signals the pass manager that the MLIR verifier need not
    // run after this pass.
    markAllAnalysesPreserved();
  }
};
} // namespace
