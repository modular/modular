//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def PackageDescription : MojoCommandDescription<
  "package", "Compile a package",
  "Translate a mojo source package into a precompiled package.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>,
  Index<3>;
