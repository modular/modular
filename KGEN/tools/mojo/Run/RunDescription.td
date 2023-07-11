//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def RunDescription : MojoCommandDescription<
  "run", "Build and execute a Mojo file",
  "Compiles the Mojo file at the given path and immediately executes it.\n"
  "\n"
  "Options for this command itself, such as the ones listed below, must appear "
  "before the input file `path` argument. Any command line arguments that "
  "appear after the Mojo source file `path` are interpreted as arguments for "
  "that Mojo program.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>,
  Index<0>;
