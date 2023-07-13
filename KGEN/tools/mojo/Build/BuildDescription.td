//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def BuildDescription : MojoCommandDescription<
  "build", "Build an executable from a Mojo file",
  "Compiles the Mojo file at the given path into an executable.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>,
  Index<1>;
