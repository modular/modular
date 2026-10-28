//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file defines the option for the bitcode libraries that a tool links
// into the LLVM IR it generates or packages into a precompiled package.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_BITCODEOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_BITCODEOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"

defm bitcode_libs : CommandOptionHidden<["--", "-"], "bitcode-libs",
  CompilationOptionGroup, Index<24>, MetaVarName<"LIBS">,
  HelpText<
    "External bitcode libraries to link into the generated LLVM IR."
  >>,
  MarshallingInfoCustom<CompilationOpts<"bitcodeLibs">,
                        "normalizeStringSmallVector">;

#endif // KGEN_TOOLS_MOJO_COMMON_BITCODEOPTIONS_TD
