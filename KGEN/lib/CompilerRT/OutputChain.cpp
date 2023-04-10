//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/OutputChain.h"
#include "LLCL/Runtime/Algorithms.h"
#include "llvm/Support/Threading.h"

using namespace M;
using namespace KGEN;

OutputChain OutputChain::copy() {
  OutputChain result(chain.copy(), loc.copy());
  result.profilerEntry = std::move(profilerEntry);
  for (const auto &ref : refs)
    result.refs.emplace_back(ref.copy());
  return result;
}

void OutputChain::await() { LLCL::await(chain); }

void OutputChain::transfer(LLCL::AnyAsyncValueRef &&argRef) {
  refs.emplace_back(std::move(argRef));
}

void OutputChain::transfer(SmallVector<LLCL::AnyAsyncValueRef> &&argRefs) {
  for (auto &argRef : argRefs)
    refs.emplace_back(std::move(argRef));
}

void OutputChain::trace(StringRef name, StringRef detail) {
  profilerEntry = MojoProfilerEntry::create(name, detail);
}

void OutputChain::markReady() {
  complete();
  // Don't consume the chain since Mojo is not copying/moving OutputChains.
  chain.copy().emplace();
}

void OutputChain::markError(StringRef message) {
  complete();
  // Don't consume the chain since Mojo is not copying/moving OutputChains.
  chain.copy().setToError({Twine(message), std::move(loc)});
}

void OutputChain::emplace() && {
  complete();
  std::move(chain).emplace();
}

void OutputChain::setToError(Error &&error) && {
  complete();
  std::move(chain).setToError({std::move(error), std::move(loc)});
}

void OutputChain::complete() {
  refs.clear();
  std::move(profilerEntry).record();
}

void OutputChain::executeAsTask(void (*resume)(int8_t *), int8_t *hdl,
                                size_t taskId) {
  // If it is present, copy the profiling entry for use by the task.
  MojoProfilerEntry taskProfilerEntry =
      profilerEntry.withNameSuffix(".task").withDetailSuffix(
          [=]() { return (Twine(" (task_id ") + Twine(taskId) + ")").str(); });
  chain.getRuntime()->getWorkQueue()->addTask(
      [taskProfilerEntry = std::move(taskProfilerEntry), resume,
       hdl]() mutable {
        taskProfilerEntry.restart();
        resume(hdl);
        std::move(taskProfilerEntry).record();
      });
}
