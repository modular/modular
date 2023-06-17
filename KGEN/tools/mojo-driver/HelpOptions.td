//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def HelpGroup : OptionGroup<"Help options.">;

def help : Flag<["--", "-"], "help">, Group<HelpGroup>, Flags<[]>,
  HelpText<"Prints help and exits.">;

def : Flag<["-"], "h">, Group<HelpGroup>, Flags<[]>,
  HelpText<"Alias for --help.">, Alias<help>;

