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
// This file defines the parser statistics option, for the subcommands that
// parse Mojo source and print the timing reports.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_PARSERSTATSOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_PARSERSTATSOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"

def parser_stats: Flag<["--", "-"], "parser-stats">,
  Group<CompilationOptionGroup>, Flags<[HelpHidden]>,
  HelpText<
    "Reports the self time the parser spends to resolve each declaration, "
    "split into its signature, its body, the synthetic signature and body of "
    "a declaration without source, and the resolution of a declaration from "
    "a precompiled package. The time of a resolution that another one "
    "demands counts only for the demanded declaration. The report takes the "
    "format and the destination of the timing reports: text on stderr, JSON "
    "with `--timing-json`, a file with `--timing-file`.">,
  MarshallingInfoFlag<CompilationOpts<"parserStats">>;

#endif // KGEN_TOOLS_MOJO_COMMON_PARSERSTATSOPTIONS_TD
