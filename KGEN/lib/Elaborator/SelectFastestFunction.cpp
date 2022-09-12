//===- SelectFastestFunction.cpp ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SelectFastestFunction.h"

#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include <numeric>

using namespace M;
using namespace KGEN;

static uint64_t benchmarkSingleFunc(CompiledFunc k, void *argMemory,
                                    void *resultMemory) {
  auto start = std::chrono::steady_clock::now();
  // Invoke the kernel.
  for (size_t i = 0; i < 10'000; ++i)
    k.invoke<void, void *, void *>(argMemory, resultMemory);

  auto stop = std::chrono::steady_clock::now();
  // Append the timing to the vector (divide by 10,000 to get the time per
  // execution)
  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
             .count() /
         10'000;
}

ErrorOr<size_t>
M::KGEN::selectFastestFunction(GeneratorInterfaceOp itf, ModuleOp primaryModule,
                               ArrayRef<FuncOp> specializations) {
  // If any of the input or result types are not OpaqueObjectInterface
  // adherents, we can't do this evaluation.
  if (llvm::any_of(llvm::concat<const Type>(itf.getArgumentTypes(),
                                            itf.getResultTypes()),
                   [](Type t) { return !t.isa<OpaqueObjectInterface>(); }))
    return Error("cannot search an interface that has a signature with types "
                 "that don't implement OpaqueObjectInterface");

  // Create the execution engine.
  auto engineOr = ExecutionEngine::create();
  if (failed(engineOr))
    return engineOr.takeError();

  ExecutionEngine engine = std::move(*engineOr);

  // Walk each configuration and generate inputs for it, then benchmark a kernel
  // for it.
  struct EvaluatedFunc {
    size_t funcIdx;
    uint64_t timing;
    uint64_t weight;
  };

  // We only want the funcs passed-in to be code-generated.
  if (auto err = engine.add(primaryModule, specializations))
    return err.takeError();

  // TODO: We should be caching these so we don't always recompute everything.
  SmallVector<EvaluatedFunc> bestPerConfig;
  for (auto cfg : llvm::make_early_inc_range(*itf.getEvalConfigs())) {
    // For these purposes we don't care about the values of the results, we just
    // need to pass a pointer in so that it doesn't segfault.
    size_t resultSize = 0;
    for (auto [type, binding] :
         llvm::zip(itf.getResultTypes(), cfg.getResultBindings())) {
      auto sizeOr = cast<OpaqueObjectInterface>(type).getSizeInBytes(
          itf->getLoc(), binding);
      if (failed(sizeOr))
        return Error("unable to allocate output space for kernel evaluation");

      resultSize += *sizeOr;
    }

    // Use std::unique_ptr here to avoid leaking memory.
    std::unique_ptr<uint8_t[]> resultMem(new uint8_t[resultSize]);

    // Get all the various sizes. Keep these as a vector because we want to
    // index into the allocated memory later and we don't want to recompute all
    // the sizes.
    SmallVector<size_t> sizes;
    for (auto [type, binding] :
         llvm::zip(itf.getArgumentTypes(), cfg.getArgBindings())) {
      auto bytesOr = cast<OpaqueObjectInterface>(type).getSizeInBytes(
          itf.getLoc(), binding);
      if (failed(bytesOr))
        return Error(
            "unable to allocate the input space for kernel evaluation");

      sizes.push_back(*bytesOr);
    }

    // Use unique_ptr to make sure we don't accidentally leak this memory.
    size_t totalMem =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    std::unique_ptr<uint8_t[]> argMem(new uint8_t[totalMem]);

    // Actually fill in the memory.
    auto memptr = (uintptr_t)argMem.get();
    for (auto [type, binding, memptrIncrement] :
         llvm::zip(itf.getArgumentTypes(), cfg.getArgBindings(), sizes)) {
      if (failed(cast<OpaqueObjectInterface>(type).populate(
              itf.getLoc(), cfg.getGenKind(), binding, (void *)memptr)))
        return Error(
            "unable to populate the input space for kernel evaluation");

      memptr += memptrIncrement;
    }

    // Evaluate each func.
    uint64_t minTiming = UINT64_MAX;
    size_t currentBest;
    for (const auto &f : llvm::enumerate(specializations)) {
      auto wrapperOr = engine.lookupOpaqueWrapper(f.value());
      if (failed(wrapperOr))
        return wrapperOr.takeError();

      uint64_t thisFuncTiming =
          benchmarkSingleFunc(*wrapperOr, argMem.get(), resultMem.get());
      // If this one is better than the previous best, use it.
      if (thisFuncTiming < minTiming) {
        minTiming = thisFuncTiming;
        currentBest = f.index();
      }
    }

    // And append the best kernel to the list.
    bestPerConfig.push_back({currentBest, minTiming, cfg.getWeight()});
  }

  // Now figure out which kernel is actually best by seeing which one has the
  // lowest timing *and* the highest config weight.
  auto best = std::min_element(
      bestPerConfig.begin(), bestPerConfig.end(),
      [](const EvaluatedFunc &lhs, const EvaluatedFunc &rhs) {
        return lhs.timing < rhs.timing && lhs.weight > rhs.weight;
      });

  // Return the best kernel.
  return best->funcIdx;
}
