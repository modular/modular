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

OutputChain OutputChain::fork() {
  // Chain and location are copied.
  OutputChain result(chain.copy(), loc.copy());
  // The 'prototype' profiler entry, references and extras can be moved.
  result.prototypeProfilerEntry = std::move(prototypeProfilerEntry);
  result.refs = std::move(refs);
  result.extras = std::move(extras);
#if MODULAR_PARANOID
  result.uses = std::move(uses);
#endif
  // The actual profiler entry is left alone.
  return result;
}

void OutputChain::await() { LLCL::await(chain); }

void OutputChain::assertReady() {
  assert(chain.isValueAvailable() &&
         "assertReady failed: output chain is not ready");
}

void OutputChain::transfer(LLCL::AnyAsyncValueRef argRef) {
  refs.emplace_back(std::move(argRef));
}

void OutputChain::transfer(SmallVector<LLCL::AnyAsyncValueRef> argRefs) {
  for (auto &argRef : argRefs)
    refs.emplace_back(std::move(argRef));
}

void OutputChain::transfer(LLCL::GenericUniquePtr extra) {
  extras.emplace_back(std::move(extra));
}

void OutputChain::trace(StringRef name, std::optional<StringRef> detail) {
  if (profilerEntry.empty()) {
    // Establish the profiling entry for this Mojo kernel call.
    profilerEntry = detail ? MojoProfilerEntry::create(name, *detail)
                           : MojoProfilerEntry::create(name);
  } else {
    // Merge the given details into the existing profile entry. This is useful
    // when we need to combine profile data contributed from both the C++
    // and Mojo sides.
    profilerEntry = detail ? profilerEntry.withNameDetailSuffix(
                                 name, [&]() { return detail->str(); })
                           : profilerEntry.withNameSuffix(name);
  }
  // (Re)establish the 'prototype' profile entry, which is only used
  // by executeAsTask() below.
  prototypeProfilerEntry = profilerEntry.copy<MojoProfilerEntry>();
}

void OutputChain::markReady() {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().emplace();
}

void OutputChain::markError(StringRef message) {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().setToError({Twine(message), std::move(loc)});
}

void OutputChain::emplace() {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().emplace();
}

void OutputChain::setToError(Error &&error) {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().setToError({std::move(error), std::move(loc)});
}

void OutputChain::recordProfilerEntry() && {
  std::move(profilerEntry).record();
}

void OutputChain::complete() {
  // IMPORTANT: Stop the profiling enry before doing any other work.
  // Even the innocent looking refs.clear() may trigger frees which can
  // be surprisingly expensive, and we don't want that to be included in
  // the kernel's time.
  std::move(profilerEntry).record();
#if MODULAR_PARANOID
  // IMPORTANT: Release uses before the refs are cleared since those refs
  // may trigger frees.
  uses.clear();
#endif
  // IMPORTANT: Clear the refs and extras before marking the output chain as
  // ready so that waiters won't see stray references, and the dtors will
  // be run before any side effects of the waiters are seen (such as
  // deleting the MGP context).
  refs.clear();
  extras.clear();
  // Record the task is done for the purposes of resource checking.
  taskIsDone();
}

void OutputChain::executeAsTask(void (*resume)(int8_t *), int8_t *hdl,
                                size_t taskId) {
  // If it is present, copy the profiling entry for use by the task.
  MojoProfilerEntry taskProfilerEntry =
      prototypeProfilerEntry.withNameSuffix(".task").withDetailSuffix(
          [=]() { return (Twine(" (task_id ") + Twine(taskId) + ")").str(); });
  chain.getRuntime()->getWorkQueue()->addTask(
      [taskProfilerEntry = std::move(taskProfilerEntry), resume,
       hdl]() mutable {
        taskProfilerEntry.restart();
        resume(hdl);
        std::move(taskProfilerEntry).record();
#if MODULAR_PARANOID
        // Sleeping here gives any await loop the chance to exit and
        // proceed while this task is still 'active'. This can trigger
        // bugs since the common case is for the task to have returned
        // all the way up to the LLCL run items loop before any emplace
        // in the task body has been acted on.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
      });
}

void OutputChain::taskIsDone() {
#if MODULAR_PARANOID
  chain.getRuntime()->getWorkQueue()->taskIsDone();
#endif
}
