//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def BuildProjectDescription : MojoCommandDescription<
  "build-project", "Builds a Mojo project.",
  "Builds a Mojo project as described by its project manifest. If a path is "
  "provided, the project at that path is built. Otherwise, builds the project "
  "in the current working directory.",
  [
    UsageInfo</*optionsName=*/"", /*inputName=*/"", /*variadicInput=*/false>,
    UsageInfo</*optionsName=*/"", /*inputName=*/"path",
             /*variadicInput=*/false>
  ]>, Index<2>;
