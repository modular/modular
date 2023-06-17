//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "HelpOptions.td"

def MojoDriverGroup : OptionGroup<
  "Options specific to the top-level driver command.">;

def version : Flag<["--", "-"], "version">, Group<MojoDriverGroup>, Flags<[]>,
  HelpText<"Prints the Mojo version and exits.">;
def : Flag<["-"], "v">, Group<MojoDriverGroup>, Flags<[]>,
  HelpText<"Alias for --version.">, Alias<version>;

