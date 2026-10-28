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
// This file defines the incremental parsing option, for the subcommands that
// parse Mojo source.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_INCRPARSINGOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_INCRPARSINGOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"

defm incr_parsing_cache_file : CommandOptionHidden<["--", "-"],
  "experimental-incr-parsing",
  CompilationOptionGroup, Index<23>, MetaVarName<"PATH">,
  HelpText<"Experimental incremental parsing backed by a module cache at "
           "the given path. The main source file is always parsed; imported "
           "modules whose sources are unchanged are served from the cache "
           "instead of being re-parsed. The cache file is created if missing "
           "and rewritten whenever any cached module changed.">>,
  MarshallingInfoString<FrontendOpts<"incrParsingCacheFile">>;

#endif // KGEN_TOOLS_MOJO_COMMON_INCRPARSINGOPTIONS_TD
