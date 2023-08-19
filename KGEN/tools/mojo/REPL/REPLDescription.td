//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def REPLDescription : MojoCommandDescription<
  "repl", "Launches the REPL.",
  "Launch a Mojo read-evaluate-print loop (REPL) that allows for interactive "
  "development.\n"
  "\n"
  "Any number of options and arguments may be specified on the command line. "
  "These are then forwarded to the underlying lldb tool, which runs the REPL.",
  [UsageInfo<"lldb-options", "", /*variadicInput=*/false>]>, Index<2>;
