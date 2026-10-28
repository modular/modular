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

include "Precompile/PrecompileDescription.td"

include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/FixItOptions.td"
include "Common/IncludeOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

defm bitcode_libs : CommandOptionHidden<["--", "-"], "bitcode-libs",
  CompilationOptionGroup, Index<4>, MetaVarName<"LIBS">,
  HelpText<
    "External bitcode libraries to package into the generated package."
  >>;

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">, HelpText<
    "Sets the path and filename for the output package. The filename must end "
    "with `.mojoc` or `.mojopkg`. The filename given here defines the "
    "package name you can then use to import the code (minus the file "
    "extension). If you don't specify this option, a `.mojoc` file is "
    "generated in the current working directory, with a name based on the name "
    "of the input directory."
  >>;

defm incr_parsing_cache_file : CommandOptionHidden<["--", "-"],
  "experimental-incr-parsing",
  OutputOptionGroup, Index<1>, MetaVarName<"PATH">,
  HelpText<"Experimental incremental parsing backed by a module cache at "
           "the given path. The main source file is always parsed; imported "
           "modules whose sources are unchanged are served from the cache "
           "instead of being re-parsed. The cache file is created if missing "
           "and rewritten whenever any cached module changed.">>,
  MarshallingInfoString<FrontendOpts<"incrParsingCacheFile">>;
