//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Telemetry.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

ScopedThread M::logToolInvocationEventAsync(
    M::Telemetry::TelemetryContext &telemetryCtx, StringRef message,
    const llvm::opt::InputArgList &args, ArrayRef<unsigned> privateArgs) {

  // TODO: The API for adding resources when initializing a telemetry context is
  // not implemented yet. We should add the current mojo version as an attribute
  // when we can.

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

  std::string s;
  llvm::raw_string_ostream ss(s);
  llvm::interleave(
      publicArgs, ss,
      [&](const llvm::opt::Arg *arg) {
        ss << StringRef(args.getArgString(arg->getIndex()));
        if (ArrayRef<const char *> values = arg->getValues(); !values.empty()) {
          ss << StringRef("=[");
          llvm::interleave(
              values, ss, [&](const char *value) { ss << StringRef(value); },
              " ");
          ss << StringRef("]");
        }
      },
      " ");
  // Notify an invocation event of the current subcommand and arguments.
  auto logger = telemetryCtx.getLogger("mojo");
  return ScopedThread(
      [logger = std::move(logger), message = message.str(), s = std::move(s)] {
        logger->emitL1Event("invoke." + message, {{"args", s}});
      });
}
