//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a "Common options" option group, as well as the "help"
// options "-h" and "--help".
//
// mojo-tblgen sorts option groups based on index; these common options are
// given a high index value so that they appear last on a list of options.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_COMMONOPTIONS_TD
#define SUPPORT_DRIVER_COMMONOPTIONS_TD

include "Support/Driver/DriverCommand.td"
include "llvm/Option/OptParser.td"

def CommonOptionGroup : OptionGroup<"Common options">, Index<99>;

def help : Flag<["--", "-"], "help">, Group<CommonOptionGroup>, Index<99>,
  HelpText<"Displays help information.">;
def h : Flag<["-"], "h">, Group<CommonOptionGroup>, Alias<help>, Index<0>;

#endif // SUPPORT_DRIVER_COMMONOPTIONS_TD
