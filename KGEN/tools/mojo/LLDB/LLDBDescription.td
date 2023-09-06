//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def LLDBDescription : MojoCommandDescription<
  "lldb", "Launches LLDB with Mojo support.",
  "Launches the LLDB debugger with support for the Mojo language."
  "\n"
  "This feature is still experimental.",
  [UsageInfo<"lldb-options", "", /*variadicInput=*/false>]>, Index<3>;
