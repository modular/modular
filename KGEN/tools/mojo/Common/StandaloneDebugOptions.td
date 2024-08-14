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
// LLDB Options
//===----------------------------------------------------------------------===//

def LLDBOptionGroup : OptionGroup<"LLDB options">, Index<1>;

defm xlldb: CommandOption<["--", "-"], "Xlldb",
  LLDBOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Passes ARG as an argument to LLDB when the debug session is "
  "launched using the LLDB command-line interface. This option can be "
  "specified multiple times. It is ignored when using the RPC mode.">>;

//===----------------------------------------------------------------------===//
// RPC Options
//===----------------------------------------------------------------------===//

def RPCOptionGroup : OptionGroup<"RPC debug server options">, Index<3>;

def rpc: Flag<["--", "-"], "rpc">,
  Group<RPCOptionGroup>, Index<1>,
  HelpText<"Launches the debug session in the external editor that has an "
  "active RPC debug server.">;

defm terminal: CommandOptionWithValues<
          ["--"], "terminal", RPCOptionGroup, Index<2>,
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
  RPCOptionGroup, Index<3>, MetaVarName<"PORT">,
  HelpText<"Uses the given PORT to communicate with the RPC debug server. "
  "Defaults to trying all ports from 12355 to 12364 inclusive.">>;

#endif // KGEN_TOOLS_MOJO_DEBUG_STANDALONEDEBUGOPTIONS_TD
