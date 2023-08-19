//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"
include "Build/BuildDescription.td"
include "Demangle/DemangleDescription.td"
include "Doc/DocDescription.td"
include "Format/FormatDescription.td"
include "Package/PackageDescription.td"
include "REPL/REPLDescription.td"
include "Run/RunDescription.td"

def : MojoDriverDescription<
  "The Mojo🔥 command line interface.",
  "This is the primary interface for Mojo's various features. When given a "
  "command argument, this launches that command. A list of all available "
  "commands appears below, and you may invoke each of them with the `--help` "
  "option to learn more about them.\n"
  "\n"
  "Alternatively, you may provide a path to a Mojo source file, in order to "
  "build and execute that file. This behavior is equivalent to the `mojo run` "
  "command; please run `mojo run --help` to learn more.\n"
  "\n"
  "Finally, invoking this program with no arguments launches the Mojo REPL. "
  "This behavior is equivalent to the `mojo repl` command; please run "
  "`mojo repl --help` to learn more.",
  [
    UsageInfo<"", "command", /*variadicInput=*/false>,
    UsageInfo<"run-options", "path", /*variadicInput=*/false>,
    UsageInfo<"options", "", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;
