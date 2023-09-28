//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DebugDescription : MojoCommandDescription<
  "debug", "Launches the LLDB debugger with support for debugging Mojo programs.",
  "Launches the LLDB debugger with support for debugging programs written in "
  "Mojo, as well as other standard languages like C and C++."
  "\n"
  "This feature is still experimental.",
  [UsageInfo<"debug-options", "", /*variadicInput=*/false>]>, Index<3>;
