//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines help text for the `greeter-cli hi` subcommand. The
// subcommand's options, on the other hand, are defined in `HiOptions.td`.
//
// The concepts used here are the same as those documented in
// `../DriverDescription.td`; read that file for more details.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  /*executable=*/"greeter-cli",
  /*subcommand=*/"hi",
  /*summary=*/"Prints \"hi\".",
  /*description=*/"Prints \"hi\" to stdout and exits. If a name is provided, "
                  "includes that name in the greeting.\n\nThis greeting is "
                  "typically used when meeting someone.",
  [
    UsageInfo</*optionsName=*/"options", /*inputName=*/"name",
              /*variadicInput=*/false>
  ]>, Index<0>;
