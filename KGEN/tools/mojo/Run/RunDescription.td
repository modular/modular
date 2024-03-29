//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def RunDescription : MojoCommandDescription<
  "run", "Builds and executes a Mojo file.",
  "Compiles the Mojo file at the given path and immediately executes it. "
  "Another way to execute this command is to simply pass a file to `mojo`. "
  "For example:\n"
  "\n"
  "    mojo hello.mojo\n"
  "\n"
  "Options for this command itself, such as the ones listed below, must appear "
  "before the input file `path` argument. Any command line arguments that "
  "appear after the Mojo source file `path` are interpreted as arguments for "
  "that Mojo program.",
  [UsageInfo<"options", "path", /*variadicInput=*/false,
             /*inNameHasArgs=*/true>]>, Index<0>;
