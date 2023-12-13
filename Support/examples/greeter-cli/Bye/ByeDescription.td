//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file is nearly equivalent to `Hi/HiDescription.td`, you can read the
// comments there for an in-depth explanation of what's going on here.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  /*executable=*/"greeter-cli",
  /*subcommand=*/"bye",
  /*summary=*/"Prints \"bye\".",
  /*description=*/"Prints \"bye\" to stdout and exits.\n\nThis greeting is "
                  "typically used when parting ways with someone.",
  [
    UsageInfo</*optionsName=*/"options", /*inputName=*/"",
              /*variadicInput=*/false>
  ]>, Index<1>;
