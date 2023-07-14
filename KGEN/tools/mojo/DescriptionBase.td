//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DRIVER_OPTIONS_BASE_TD
#define MOJO_DRIVER_OPTIONS_BASE_TD

include "Support/Driver/DriverCommand.td"

// A description of the Mojo top-level driver executable.
class MojoDriverDescription<string summary, string description,
                            list<UsageInfo> usages> :
  DriverDescription</*executable=*/"mojo", summary, description, usages>;

// A description of a Mojo driver subcommand.
class MojoCommandDescription<string subcommand, string summary,
                             string description, list<UsageInfo> usages> :
  CommandDescription</*executable=*/"mojo", subcommand, summary, description,
                     usages>;

#endif // MOJO_DRIVER_OPTIONS_BASE_TD
