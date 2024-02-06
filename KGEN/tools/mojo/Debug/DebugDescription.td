//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DebugDescription : MojoCommandDescription<
  "debug", "Launches the Mojo debugger using the command-line interface or an "
  "external editor.",
  "This command, which underneath uses the LLDB debugger, offers four basic "
  "debug session modes:\n\n"
  "1. Build and debug a Mojo file.\n\n"
  "    mojo debug [options] <file.mojo> [runtime args]\n\n"
  "Builds the Mojo file at the given path and immediately launches it under the"
  " debugger. Options, which come before the Mojo file, can include any "
  "compilation options expected by the `mojo run`, as well as regular debugging"
  "commands. Runtime args, which come after the Mojo file, are passed "
  "directly to the debuggee upon launch. By default, this mode uses -O0 and "
  "--debug-level=full as compilation options.\n\n"
  "2. Debug a precompiled program.\n\n"
  "    mojo debug [options] <program> [runtime args]\n\n"
  "Launches the program at the given path under the debugger. Options, which "
  "come before the program path, cannot include compilation commands. Runtime "
  "args, which come after the program path, are passed directly to the debuggee"
  " upon launch.\n\n"
  "3. Attach to a running process.\n\n"
  "    mojo debug [options] [--pid <pid> | --process-name <process-name>]\n\n"
  "Attaches to the process given by its pid or name. Options, which come before"
  " the process identifier, cannot include compilation commands.\n\n"
  "4. Start the LLDB command-line interface.\n\n"
  "    mojo debug [options]\n\n"
  "Launches the LLDB CLI with support for debugging Mojo programs. This command"
  " only supports LLDB options (via the --Xlldb option).\n\n"
  "Besides this, two interfaces can be used for the debug session:\n\n"
  "- LLDB CLI: by default, all debug session modes are launched using the "
  "regular LLDB command-line interface.\n\n"
  "- External editor: if the --rpc option is provided, the debug session is "
  "launched in a editor that has configured an RPC debug server, e.g. VS Code "
  "via the RPC server launched by the Mojo extension.\n\n"
  "Finally, it is worth mentioning that this debugger can debug programs "
  "written in other standard native languages like C and C++, as it is based "
  "on LLDB.",
  [UsageInfo<"debug-options", "", /*variadicInput=*/false>]>, Index<3>;
