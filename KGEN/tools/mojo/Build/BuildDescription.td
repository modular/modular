//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def BuildDescription : MojoCommandDescription<
  "build", "Builds an executable from a Mojo file.",
  "Compiles the Mojo file at the given path into an executable.\n\n"
  "By default, the executable is saved to the current directory and named the "
  "same as the input file, but without a file extension.\n\n"
  "Beware that any Python libraries used in your Mojo project are not included "
  "in the executable binary, so they must be provided by the environment "
  "where you run the executable.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<1>;
