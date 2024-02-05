//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Debug/DebugDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"
include "../Common/TargetOptions.td"


//===----------------------------------------------------------------------===//
// LLDB Options
//===----------------------------------------------------------------------===//

def LLDBOptionGroup : OptionGroup<"LLDB options">, Index<1>;

defm xlldb: CommandOption<["--", "-"], "Xlldb",
  LLDBOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Passes ARG as an argument to LLDB when the debug session is "
  "launched using the LLDB command line interface. This option can be "
  "specified multiple times.">>;

//===----------------------------------------------------------------------===//
// Attach Options
//===----------------------------------------------------------------------===//

def AttachOptionGroup : OptionGroup<"Attach options">, Index<2>;

defm pid: CommandOption<["--", "-"], "pid",
  AttachOptionGroup, Index<1>, MetaVarName<"PID">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "PID.">>;

defm name: CommandOption<["--", "-"], "name",
  AttachOptionGroup, Index<2>, MetaVarName<"NAME">,
  HelpText<"Indicates the debugger to attach to the process with the given "
  "name.">>;

//===----------------------------------------------------------------------===//
// RPC Options
//===----------------------------------------------------------------------===//

def RPCOptionGroup : OptionGroup<"RPC debug server options">, Index<3>;

def rpc: Flag<["--", "-"], "rpc">,
  Group<RPCOptionGroup>, Index<1>,
  HelpText<"Launches the debug session in the external editor that has an "
  "active RPC debug server.">;

defm port: CommandOption<["--", "-"], "port",
  RPCOptionGroup, Index<2>, MetaVarName<"PORT">,
  HelpText<"Uses the given PORT to communicate with the RPC debug server. "
  "Defaults to 12345.">>;

defm secret: CommandOption<["--", "-"], "secret",
  RPCOptionGroup, Index<3>, MetaVarName<"SECRET">,
  HelpText<"Uses the given SECRET string as a minimal authentication mechanism "
  "when communicating with the RPC debug server. It must be used only if the "
  "RPC debug server expects it.">>;
