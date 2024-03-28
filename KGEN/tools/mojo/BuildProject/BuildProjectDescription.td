//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def BuildProjectDescription : MojoCommandDescription<
  "build-project", "Builds a Mojo project.",
  "Builds a Mojo project as described by its project manifest.",
  [UsageInfo</*optionsName=*/"", /*inputName=*/"", /*variadicInput=*/false>]>,
  Index<2>;
