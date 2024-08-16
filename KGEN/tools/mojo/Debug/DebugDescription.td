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
  "* Build and debug a Mojo file.\n\n"
  "        mojo debug [options] <file.mojo> [runtime args]\n\n"
  "    Builds the Mojo file at the given path and launches it under the"
  " debugger. Options, which come before the Mojo file, can include any "
  "compilation options expected by the `mojo run`, as well as regular debugging"
  "commands. Runtime args, which come after the Mojo file, are passed "
  "directly to the debuggee upon launch. By default, this mode uses `-O0` and "
  "`--debug-level=full` as compilation options.\n\n"
  "* Debug a precompiled program.\n\n"
  "        mojo debug [options] <program> [runtime args]\n\n"
  "    Launches the program at the given path in the debugger. Options, which "
  "come before the program path, cannot include compilation commands. Runtime "
  "args, which come after the program path, are passed directly to the debuggee"
  " upon launch.\n\n"
  "* Attach to a running process.\n\n"
  "        mojo debug [options] [--pid <pid> | --process-name <process-name>]\n\n"
  "    Attaches to the process specified by pid or name, which can be the full "
  "path of the process' executable. Options other than the process identifier "
  "cannot include compilation options.\n\n"
  "* Start the LLDB command-line interface.\n\n"
  "        mojo debug [options]\n\n"
  "    Launches the LLDB CLI with support for debugging Mojo programs. This"
  " command only supports LLDB options (via the `--X` option).\n\n"
  "You can also select one of two interfaces for the debug session:\n\n"
  "* LLDB CLI: By default, all debug session modes are launched using the "
  "regular LLDB command-line interface.\n\n"
  "* RPC debug server: If you add the `--rpc` option, the debug session is "
  "launched in a editor that has spawned an RPC debug server, e.g. VS Code via "
  "the RPC debug server launched by the Mojo extension. Besides that, the "
  "environment variable and the current working directory of this invocation "
  "are preserved when launching programs on the RPC debug server side.\n\n"
  "Finally, it is worth mentioning that this debugger can debug programs "
  "written in other standard native languages like C and C++, as it is based "
  "on LLDB.",
  [UsageInfo<"debug-options", "", /*variadicInput=*/false>]>, Index<4>;
