//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def BuildDescription : MojoCommandDescription<
  "build", "Build an executable from a Mojo file",
  "Compiles the Mojo file at the given path into an executable.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<1>;
