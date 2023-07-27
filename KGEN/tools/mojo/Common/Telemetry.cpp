//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Telemetry.h"
#include "llvm/Option/ArgList.h"

using namespace M;

void M::initializeTelemetry(
    LLCL::RCRef<LLCL::Telemetry::TelemetryContext> telemetryCtx,
    const State &state, const llvm::opt::InputArgList &args,
    ArrayRef<unsigned> privateArgs) {
  if (!telemetryCtx)
    return;

  // TODO: The API for adding resources when initializing a telemetry context is
  // not implemented yet. We should add the current mojo version as an attribute
  // when we can.

  // Notify an invocation event of the current subcommand and arguments.
  LLCL::Telemetry::Logs::Logger::LogStream os =
      telemetryCtx->getLogger("mojo")->getInfo("invoke." +
                                               StringRef(state.subcommand));

  // Extract the recordable arguments from the command line, and order them by
  // id to ensure a deterministic order.
  DenseSet<unsigned> privateArgsSet(privateArgs.begin(), privateArgs.end());
  SmallVector<const llvm::opt::Arg *> publicArgs(
      llvm::make_filter_range(args.getArgs(), [&](const llvm::opt::Arg *arg) {
        return arg->getOption().getKind() != llvm::opt::Option::InputClass &&
               !privateArgsSet.count(arg->getOption().getID());
      }));
  llvm::stable_sort(publicArgs, [](const auto *lhs, const auto *rhs) {
    return lhs->getOption().getID() < rhs->getOption().getID();
  });
  llvm::interleave(
      publicArgs, os,
      [&](const auto *arg) {
        os << StringRef(args.getArgString(arg->getIndex()));
      },
      " ");
}
