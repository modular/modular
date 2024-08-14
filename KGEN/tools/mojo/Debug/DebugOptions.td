//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Debug/DebugDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"
include "../Common/DiagnosticOptions.td"
include "../Common/TargetOptions.td"


//===----------------------------------------------------------------------===//
// LLDB Options
//===----------------------------------------------------------------------===//

def LLDBOptionGroup : OptionGroup<"LLDB options">, Index<1>;

defm xlldb: CommandOption<["--", "-"], "Xlldb",
  LLDBOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Passes ARG as an argument to LLDB when the debug session is "
  "launched using the LLDB command-line interface. This option can be "
  "specified multiple times. It is ignored when using the RPC mode.">>;

// As this option is hidden, it needs to exist within a non-empty option group,
// so we put it here just because there's no place better.
def dry_run: Flag<["--", "-"], "dry-run">,
  Group<LLDBOptionGroup>, Index<10>,
  Flags<[HelpHidden]>,
  HelpText<"Instead of executing LLDB on the CLI, dumps the command that would "
  "be used to start the LLDB command-line interface. If using the RPC mode, "
  "dumps the payload that would be sent to the RPC server.">;

//===----------------------------------------------------------------------===//
// Attach Options
//===----------------------------------------------------------------------===//

def AttachOptionGroup : OptionGroup<"Attach options">, Index<2>;

defm pid: CommandOption<["--", "-"], "pid",
  AttachOptionGroup, Index<1>, MetaVarName<"PID">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "PID.">>;

defm process_name: CommandOption<["--", "-"], "process-name",
  AttachOptionGroup, Index<2>, MetaVarName<"NAME">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "name or path.">>;

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
