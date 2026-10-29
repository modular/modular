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

include "Doc/DocDescription.td"

include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/IncludeOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Sets the path and filename for the JSON output. If not provided, "
           "output is written to stdout.">>;

defm docs_base_path : CommandOption<["--", "-"], "docs-base-path",
  OutputOptionGroup, Index<1>, MetaVarName<"PATH">,
  HelpText<"Sets the path prefix for generated documentation links.">>,
  MarshallingInfoString<FrontendOpts<"docsBasePath">>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def validate_doc_strings : Flag<["--", "-"], "validate-doc-strings">,
  Group<DiagnosticOptionGroup>, Index<15>,
  Flags<[HelpHidden]>,
  HelpText<"Deprecated: use -Werror instead. Treat doc string warnings as errors.">;
