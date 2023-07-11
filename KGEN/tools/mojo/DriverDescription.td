//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"
include "Demangle/DemangleDescription.td"
include "Doc/DocDescription.td"
include "Package/PackageDescription.td"
include "REPL/REPLDescription.td"
include "Run/RunDescription.td"

def : MojoDriverDescription<
  "The Mojo🔥 command line interface", "Interface with Mojo's various features.",
  /*metaVarName=*/"command", /*requiresInput=*/false, /*variadicInput=*/false>;
