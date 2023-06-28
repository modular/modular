//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DRIVER_OPTIONS_BASE_TD
#define MOJO_DRIVER_OPTIONS_BASE_TD

include "KGEN/DriverCommand.td"

// A description of the Mojo top-level driver executable.
class MojoDriverDescription<
  string summary, string description, string inputMetaVarName,
  bit requiresInput, bit variadicInput>
    : DriverDescription</*executable=*/"mojo-driver", summary, description,
                        inputMetaVarName, requiresInput, variadicInput>;

// A description of a Mojo driver subcommand.
class MojoCommandDescription<
  string subcommand, string summary, string description,
  string inputMetaVarName, bit requiresInput, bit variadicInput>
    : CommandDescription</*executable=*/"mojo-driver", subcommand, summary,
                         description, inputMetaVarName, requiresInput,
                         variadicInput>;

#endif // MOJO_DRIVER_OPTIONS_BASE_TD
