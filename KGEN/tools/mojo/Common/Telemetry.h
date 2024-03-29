//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H
#define KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H

#include "Support/Driver/DriverSupport.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Option/OptTable.h"

namespace M {
/// Initialize a telemetry context for the given state and arguments. An
/// additional set of "private" arguments can be provided, which will be
/// redacted from telemetry events. By default, all input arguments are
/// private.
void initializeTelemetry(M::Telemetry::TelemetryContext &telemetryCtx,
                         StringRef message, const llvm::opt::InputArgList &args,
                         ArrayRef<unsigned> privateArgs = {});
} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H
