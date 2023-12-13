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

OutputChain::~OutputChain() { complete(); }

OutputChain OutputChain::fork() {
  assert(!cudaStream && "cant fork for CUDA kernel launches");
  // Chain and location are copied.
  OutputChain result(chain.copy(), loc.copy());
  // References and extras are moved.
  result.refs = std::move(refs);
  refs.clear();
  result.extras = std::move(extras);
  extras.clear();
#if MODULAR_PARANOID
  result.uses = std::move(uses);
  uses.clear();
#endif
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

void OutputChain::complete() {
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

void OutputChain::taskIsDone() {
#if MODULAR_PARANOID
  chain.getRuntime()->getWorkQueue()->taskIsDone();
#endif
}
