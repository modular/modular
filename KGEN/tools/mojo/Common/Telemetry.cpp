//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Telemetry.h"

using namespace M;

void M::initializeTelemetry(
    LLCL::RCRef<Telemetry::TelemetryContext> telemetryCtx, const State &state,
    const llvm::opt::InputArgList &args,
    ArrayRef<llvm::opt::OptSpecifier> privateArgs) {
  if (!telemetryCtx)
    return;

  // Notify an invocation event with the subcommand, arguments, and the current
  // mojo version.
  // TODO: The API for adding attributes to the invocation event is not
  // implemented yet. We should add the subcommand, arguments, timestamp, and
  // the current mojo version as attributes to the invocation event.
}
