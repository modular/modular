//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"
include "Build/BuildDescription.td"
include "Debug/DebugDescription.td"
include "Demangle/DemangleDescription.td"
include "Doc/DocDescription.td"
include "Format/FormatDescription.td"
include "Package/PackageDescription.td"
include "REPL/REPLDescription.td"
include "Run/RunDescription.td"
include "Test/TestDescription.td"

def : MojoDriverDescription<
  "The Mojo🔥 command line interface.",
  "The `mojo` CLI provides all the tools you need for Mojo development, such "
  "as commands to run, compile, and package Mojo code. A list of all commands "
  "are listed below, and you can learn more about each one by adding the "
  "`--help` option to the command (for example, `mojo package --help`).\n"
  "\n"
  "However, you may omit the `run` and `repl` commands. That is, you can run "
  "a Mojo file by simply passing the filename to `mojo`:\n"
  "\n"
  "    mojo hello.mojo\n"
  "\n"
  "And you can start a REPL session by running `mojo` with no commands.\n"
  "\n"
  "To update Mojo to the latest version, use the "
  "[`modular` tool](https://docs.modular.com/cli/):\n"
  "\n"
  "    modular update mojo\n"
  "\n"
  "You can check your current version with `mojo --version`. "
  "For information about Mojo updates, see the "
  "[Mojo changelog](https://docs.modular.com/mojo/changelog.html).",
  [
    UsageInfo<"", "command", /*variadicInput=*/false>,
    UsageInfo<"run-options", "path", /*variadicInput=*/false>,
    UsageInfo<"options", "", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;
