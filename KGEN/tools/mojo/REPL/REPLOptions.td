//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "../OptionsBase.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// repl Command
//===----------------------------------------------------------------------===//

def REPLDescription : MojoCommandDescription<
  "repl", "Launch the REPL",
  "Launch a Mojo read-evaluate-print loop that allows for interactive "
  "development.\n"
  "\n"
  "Any number of options and arguments may be specified on the command line. "
  "These are then forwarded to the underlying lldb tool, which runs the REPL.",
  /*inputMetaVarName=*/"lldb-args", /*requiresInput=*/false,
  /*variadicInput=*/true>;
