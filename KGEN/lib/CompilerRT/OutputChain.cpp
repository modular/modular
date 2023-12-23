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

void OutputChain::markReady() {
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

void OutputChain::setToError(Error &&error) {
  // CAUTION: Must copy so chain remains valid.
  chain.copy().setToError({std::move(error), std::move(loc)});
}
