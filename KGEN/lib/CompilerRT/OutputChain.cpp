//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/OutputChain.h"
#include "CUDASupport/CUDAAsyncValue.h"
#include "LLCL/Runtime/Algorithms.h"
#include "llvm/Support/Threading.h"

using namespace M;
using namespace KGEN;

OutputChain OutputChain::fork() {
  // Chain, location and parent are copied.
  OutputChain result(chain.copy(), loc.copy());
  result.parentEventId = parentEventId;
  // References and extras are moved.
  result.refs = std::move(refs);
  result.extras = std::move(extras);
#if MODULAR_PARANOID
  result.uses = std::move(uses);
#endif
  // The profiler entries are left alone.
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
  if constexpr (MojoProfilerEntry::isEnabled()) {
    const auto &entry = profilerEntries.emplace_back(
        detail ? MojoProfilerEntry::create(name, *detail)
               : MojoProfilerEntry::create(name));
    parentEventId = entry.getId();
  }
}

void OutputChain::trace(StringRef name,
                        llvm::function_ref<std::string()> detailFn) {
  if constexpr (MojoProfilerEntry::isEnabled()) {
    const auto &entry =
        profilerEntries.emplace_back(MojoProfilerEntry::create(name, detailFn));
    parentEventId = entry.getId();
  }
}

void OutputChain::trace(InternableString name) {
  if constexpr (MojoProfilerEntry::isEnabled()) {
    const auto &entry =
        profilerEntries.emplace_back(MojoProfilerEntry::create(name));
    parentEventId = entry.getId();
  }
}

void OutputChain::markReady() {
  complete();
  // CAUTION: Must copy so chain remains valid.
  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // There's currently no 'regular' chain to emplace for kernels launched via
  // cuda.kernel.execute.via_cpu, and the CPU portion is expected to be
  // synchronous. However, there are various mark_ready calls scattered about
  // as part of the single_thread_blocking_override handling. So just silently
  // ignore those.
  if (chain.isType<Chain>())
    chain.copy().emplace<Chain>();
}

void OutputChain::markError(StringRef message) {
  complete();
  if (chain.isError()) {
    // Currently, fused Mojo kernels may end up calling mark_error more than
    // once for the same kernel call. Rather than assert failing, which causes
    // the root error to be lost, instead forgive the subsequent errors.
    // However, note we're on thin ice here since the above complete() will have
    // already released all resources for the call.
    // TODO(#25740): Remove once fused kernels correctly early exit.
    llvm::errs() << "Mojo kernel has attempted to mark_error more than once, "
                    "with message '"
                 << message << "'\n";
    return;
  }
  // CAUTION: Must copy so chain remains valid.
  chain.copy().setToError({Twine(message), std::move(loc)});
}

void OutputChain::emplace() {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().emplace<Chain>();
}

void OutputChain::setToError(Error &&error) {
  complete();
  // CAUTION: Must copy so chain remains valid.
  chain.copy().setToError({std::move(error), std::move(loc)});
}

void OutputChain::recordProfilerEntries() && {
  for (auto &entry : profilerEntries)
    std::move(entry).record();
  profilerEntries.clear();
}

void OutputChain::complete() {
  // IMPORTANT: Stop the profiling entries before doing any other work.
  // Even the innocent looking refs.clear() may trigger frees which can
  // be surprisingly expensive, and we don't want that to be included in
  // the kernel's time.
  for (auto &entry : profilerEntries)
    std::move(entry).record();
  profilerEntries.clear();
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
                                size_t taskId, bool useGlobalQueue) {
  chain.getRuntime()->getWorkQueue()->addTask(
      [parentId = this->parentEventId, taskId, resume, hdl]() mutable {
        // Use the 'prototype' profiling entry, but augment with the task id.
        TimeTraceScope scope(MojoProfilerEntry::createWithParent(
            parentId, StringLiteral("task"), (uint64_t)taskId));
        resume(hdl);
#if MODULAR_PARANOID
        // Sleeping here gives any await loop the chance to exit and
        // proceed while this task is still 'active'. This can trigger
        // bugs since the common case is for the task to have returned
        // all the way up to the LLCL run items loop before any emplace
        // in the task body has been acted on.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
      },
      useGlobalQueue ? -1 : (int)taskId);
}

void OutputChain::taskIsDone() {
#if MODULAR_PARANOID
  chain.getRuntime()->getWorkQueue()->taskIsDone();
#endif
}
