//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SelectFastestFunction.h"

#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/MicroBenchmark.h"
#include "llvm/Support/Debug.h"

#include <chrono>
#include <numeric>

#define DEBUG_TYPE "select-fastest-function"

using namespace M;
using namespace KGEN;
using namespace std::chrono_literals;

static uint64_t benchmarkSingleFunc(CompiledFunc k, void *argMemory,
                                    void *resultMemory) {
  MicroBenchmark benchmark(
      "kgen benchmark function", [&](MicroBenchmark::State &state) {
        for (auto _ : state)
          k.invoke<void, void *, void *>(argMemory, resultMemory);
      });
  // Run the benchmark for at most 20ms when building Modular in debug model
  // (and you do not really care about performance) and 100ms in release mode.
  //
  // TODO: This should be configurable by the user.
  MicroBenchmark::RunOptions runOptions;
  runOptions.printWarningIfDebugMode = false;
#ifdef MODULAR_DEBUG
  runOptions.minRuntime = 20ms;
#else  // MODULAR_DEBUG
  runOptions.minRuntime = 100ms;
#endif // MODULAR_DEBUG

  // Benchmark the function.
  (void)benchmark.run(runOptions);

  // Get the trimmed mean time in nanoseconds.
  auto time =
      benchmark.measurement(MicroBenchmark::ReportMetric::kTrimmedMeanLatency,
                            MicroBenchmark::TimeUnit::kNanoseconds);
  return std::lround(time);
}

ErrorOr<size_t>
M::KGEN::selectFastestFunction(GeneratorInterfaceOp itf, SymbolTable &symtab,
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

  // Make all the specializations public for now.
  SmallVector<LinkageAttr> linkage;
  for (auto s : specializations) {
    linkage.push_back(s.getLinkageAttr());
    s.setLinkageAttr(LinkageAttr::get(itf.getContext(), Linkage::Public));
  }

  // We only want the funcs passed-in to be code-generated.
  if (ErrorOrSuccess err =
          engine.add(cast<ModuleOp>(symtab.getOp()), specializations))
    return err.takeError();

  // And now reset them. We have to be explicit cause otherwise zip adds a const
  // we don't want here.
  for (std::tuple<FuncOp, LinkageAttr> symAndLink :
       llvm::zip(specializations, linkage))
    std::get<0>(symAndLink).setLinkageAttr(std::get<1>(symAndLink));

  // TODO: We should be caching these so we don't always recompute everything.
  SmallVector<EvaluatedFunc> bestPerConfig;
  for (auto cfg : llvm::make_early_inc_range(*itf.getEvalConfigs())) {
    // Create pointers for each result. We'll use this for comparing the outputs
    // against each other.
    SmallVector<size_t> resultSizes;
    resultSizes.reserve(itf.getNumResults());
    for (auto [type, binding] :
         llvm::zip(itf.getResultTypes(), cfg.getResultBindings())) {
      auto sizeOr = cast<OpaqueObjectInterface>(type).getSizeInBytes(
          itf->getLoc(), binding);
      if (failed(sizeOr))
        return Error("unable to allocate output space for kernel evaluation");

      resultSizes.push_back(*sizeOr);
    }

    // Use std::unique_ptr here to avoid leaking memory.
    size_t resultSize = std::accumulate(resultSizes.begin(), resultSizes.end(),
                                        1, std::multiplies<>());
    std::unique_ptr<uint8_t[]> resultMem(new uint8_t[resultSize]);

    // Get all the various sizes. Keep these as a vector because we want to
    // index into the allocated memory later and we don't want to recompute all
    // the sizes.
    SmallVector<size_t> argSizes;
    argSizes.reserve(itf.getNumArguments());
    for (auto [type, binding] :
         llvm::zip(itf.getArgumentTypes(), cfg.getArgBindings())) {
      auto bytesOr = cast<OpaqueObjectInterface>(type).getSizeInBytes(
          itf.getLoc(), binding);
      if (failed(bytesOr))
        return Error(
            "unable to allocate the input space for kernel evaluation");

      argSizes.push_back(*bytesOr);
    }

    // Use unique_ptr to make sure we don't accidentally leak this memory.
    size_t totalMem = std::accumulate(argSizes.begin(), argSizes.end(), 1,
                                      std::multiplies<>());
    std::unique_ptr<uint8_t[]> argMem(new uint8_t[totalMem]);

    // Actually fill in the memory.
    auto memptr = (uintptr_t)argMem.get();
    for (auto [type, binding, memptrIncrement] :
         llvm::zip(itf.getArgumentTypes(), cfg.getArgBindings(), argSizes)) {
      if (failed(cast<OpaqueObjectInterface>(type).populate(
              itf.getLoc(), cfg.getGenKind(), binding, (void *)memptr)))
        return Error(
            "unable to populate the input space for kernel evaluation");

      memptr += memptrIncrement;
    }

    // Evaluate each func.
    uint64_t minTiming = UINT64_MAX;
    size_t currentBest = 0;
    std::unique_ptr<uint8_t[]> prevResultMem(new uint8_t[resultSize]);
    bool ranAtLeastOnce = false;
    auto evaluateFunction = [&](FuncOp func, size_t idx) -> ErrorOrSuccess {
      auto wrapperOr = engine.lookupOpaqueWrapper(func.getName(), func);
      if (failed(wrapperOr))
        return wrapperOr.takeError();

      // Run the function once.
      wrapperOr->invoke<void, void *, void *>(argMem.get(), resultMem.get());

      // If we have previous tags (i.e. not the first run) then compare the
      // results of this run with the previous run. The outputs must be equal to
      // whatever tolerance the user specifies in their implementation of the
      // TypeInterface.
      if (ranAtLeastOnce) {
        auto lhsMemPtr = (uintptr_t)prevResultMem.get();
        auto rhsMemPtr = (uintptr_t)resultMem.get();
        for (auto [type, binding, memptrIncrement] : llvm::zip(
                 func.getResultTypes(), cfg.getResultBindings(), resultSizes)) {
          auto equalsOr = cast<OpaqueObjectInterface>(type).equals(
              func.getLoc(), binding, (void *)lhsMemPtr, (void *)rhsMemPtr);
          if (failed(equalsOr))
            return Error("could not compare outputs of the function with a "
                         "previous run");

          if (!*equalsOr)
            return Error("function did not sufficiently match a previous run");

          lhsMemPtr += memptrIncrement;
          rhsMemPtr += memptrIncrement;
          return success();
        }
      }

      // Update the result memory to prepare for the next run.
      ranAtLeastOnce = true;
      memcpy(prevResultMem.get(), resultMem.get(), resultSize);

      // Now run the function.
      uint64_t thisFuncTiming =
          benchmarkSingleFunc(*wrapperOr, argMem.get(), resultMem.get());
      LLVM_DEBUG({
        llvm::dbgs() << "Timing: " << thisFuncTiming
                     << " (ns) for configuration: " << cfg << " for func:\n";
        func.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      // If this one is better than the previous best, use it.
      if (thisFuncTiming < minTiming) {
        minTiming = thisFuncTiming;
        currentBest = idx;
      }

      return success();
    };

    for (const auto &f : llvm::enumerate(specializations))
      if (auto err = evaluateFunction(f.value(), f.index()))
        return err.takeError();

    // And append the best kernel to the list.
    bestPerConfig.push_back({currentBest, minTiming, cfg.getWeight()});

    // Finally, free any memory allocated by any of the interface types.
    auto resultPtr = (uintptr_t)resultMem.get();
    for (auto [type, binding, memptrIncrement] : llvm::zip(
             itf.getResultTypes(), cfg.getResultBindings(), resultSizes)) {
      cast<OpaqueObjectInterface>(type).destroy(binding, (void *)resultPtr);
      resultPtr += memptrIncrement;
    }
    auto argPtr = (uintptr_t)argMem.get();
    for (auto [type, binding, memptrIncrement] :
         llvm::zip(itf.getArgumentTypes(), cfg.getArgBindings(), argSizes)) {
      cast<OpaqueObjectInterface>(type).destroy(binding, (void *)argPtr);
      argPtr += memptrIncrement;
    }
  }

  // Now figure out which kernel is actually best by seeing which one has the
  // lowest timing *and* the highest config weight.
  auto best = std::min_element(
      bestPerConfig.begin(), bestPerConfig.end(),
      [](const EvaluatedFunc &lhs, const EvaluatedFunc &rhs) {
        return lhs.timing < rhs.timing && lhs.weight > rhs.weight;
      });

  LLVM_DEBUG({
    llvm::dbgs() << "Fastest implementation:\n";
    specializations[best->funcIdx]->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Return the best kernel.
  return best->funcIdx;
}

ErrorOr<size_t>
M::KGEN::evaluateSpecializations(FuncOp evaluator, SymbolTable &symtab,
                                 ArrayRef<FuncOp> specializations) {
  // Create the execution engine.
  auto engineOr = ExecutionEngine::create();
  if (failed(engineOr))
    return engineOr.takeError();
  ExecutionEngine engine = std::move(*engineOr);

  // Make all the specializations public for now.
  auto publicLinkage =
      LinkageAttr::get(evaluator.getContext(), Linkage::Public);
  SmallVector<LinkageAttr> origLinkages;
  LinkageAttr evaluatorLinkage = evaluator.getLinkageAttr();

  evaluator.setLinkageAttr(publicLinkage);
  for (auto s : specializations) {
    origLinkages.push_back(s.getLinkageAttr());
    s.setLinkageAttr(publicLinkage);
  }

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);
  if (ErrorOrSuccess err =
          engine.add(cast<ModuleOp>(symtab.getOp()), funcsToCompile))
    return err.takeError();

  // Reset the linkages.
  evaluator.setLinkageAttr(evaluatorLinkage);
  for (auto [func, linkage] : llvm::zip(specializations, origLinkages))
    const_cast<FuncOp *>(&func)->setLinkageAttr(linkage);

  // Get pointers to all the candidates.
  SmallVector<void *> candidatePtrs;
  for (FuncOp candidate : specializations) {
    ErrorOr<CompiledFunc> func =
        engine.lookup(candidate.getSymName(), candidate);
    if (func.isError())
      return func.takeError();

    candidatePtrs.push_back(func->getFunctionPointer());
  }

  // Lookup the evaluator function
  ErrorOr<CompiledFunc> evaluatorFunc =
      engine.lookup(evaluator.getSymName(), evaluator);
  if (evaluatorFunc.isError())
    return evaluatorFunc.takeError();

  // Invoke the evaluator.
  ssize_t bestIdx = evaluatorFunc->invoke<ssize_t, void **, ssize_t>(
      candidatePtrs.data(), candidatePtrs.size());
  if (bestIdx == -1)
    return Error("user-provided evaluator returned failure");
  if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= candidatePtrs.size())
    return Error("user-provided evaluator returned an erroneous result");

  LLVM_DEBUG({
    llvm::dbgs() << "Fastest implementation:\n";
    specializations[bestIdx]->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Return the best kernel.
  return bestIdx;
}
