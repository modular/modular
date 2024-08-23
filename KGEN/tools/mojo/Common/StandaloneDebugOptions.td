//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines debug options that apply across both the `mojo debug` and
// `mojo test` commands.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD
#define KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Debugger Options
//===----------------------------------------------------------------------===//

def DebuggerOptionGroup : OptionGroup<"Debugger options">, Index<11>;

defm xdebugger: CommandOption<["--", "-"], "X",
  DebuggerOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Passes ARG as an argument to the debugger when the debug session "
  "is launched using the debugger command-line interface. This option can be "
  "specified multiple times. It is ignored when using the RPC mode.">>;

//===----------------------------------------------------------------------===//
// Debug Server Options
//===----------------------------------------------------------------------===//

def DebugServerOptionGroup : OptionGroup<"Debug server options">, Index<12>;

def vscode: Flag<["--", "-"], "vscode">,
  Group<DebugServerOptionGroup>, Index<1>,
  HelpText<"Launches the debug session on VS Code via the Mojo extension.">;

def rpc: Flag<["--", "-"], "rpc">,
  Group<DebugServerOptionGroup>, Index<1>,
  HelpText<"Alias for --vscode.">;

defm terminal: CommandOptionWithValues<
          ["--"], "terminal", DebugServerOptionGroup, Index<2>,
          MetaVarName<"TERMINAL">,
          HelpText<"The type of terminal to use when starting a launch debug "
                   "session.\n\n"
                   "* `console` (default): the debuggee will be launched in the"
                   " default environment for the editor. If using VS Code, this"
                   " will be the Debug Console.\n"
                   "* `dedicated`: the debuggee will be launched in a dedicated"
                   " terminal within the editor.">,
          Values<"console,integrated">>;

defm port: CommandOption<["--", "-"], "port",
  DebugServerOptionGroup, Index<3>, MetaVarName<"PORT">,
  HelpText<"Uses the given PORT to communicate with the RPC debug server. "
  "Defaults to trying all ports from 12355 to 12364 inclusive.">>;

#endif // KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD
