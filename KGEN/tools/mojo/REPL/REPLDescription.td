//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def REPLDescription : MojoCommandDescription<
  "repl", "Launches the Mojo REPL.",
  "Launches a Mojo read-evaluate-print loop (REPL) environment, which provides "
  "interactive development in the terminal. You can also start the REPL by "
  "simply running `mojo`.\n"
  "\n"
  "Any number of options and arguments may be specified on the command line. "
  "These are then forwarded to the underlying lldb tool, which runs the REPL.",
  [UsageInfo<"lldb-options", "", /*variadicInput=*/false>]>, Index<3>;
