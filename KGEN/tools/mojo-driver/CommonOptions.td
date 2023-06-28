//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a "Common Options" option group, as well as the "help"
// options "-h" and "--help".
//
// mojo-tblgen sorts option groups based on index; these common options are
// given a high index value so that they appear last on a list of options.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DRIVER_COMMONOPTIONS_TD
#define MOJO_DRIVER_COMMONOPTIONS_TD

include "Support/Option/OptParser.td"

def CommonOptionGroup : OptionGroup<"Common Options">, Index<99>;

def help : Flag<["--", "-"], "help">, Group<CommonOptionGroup>, Index<0>,
  Flags<[]>, HelpText<"Prints help information.">;

def h : Flag<["-"], "h">, Group<CommonOptionGroup>, Index<0>, Flags<[]>,
  HelpText<"Alias for --help.">, Alias<help>;

#endif // MOJO_DRIVER_COMMONOPTIONS_TD
