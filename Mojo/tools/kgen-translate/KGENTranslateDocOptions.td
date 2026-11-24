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

// The options of `mojo doc`.
include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/IncludeOptions.td"
include "KGENTranslateCommonOptions.td"

def KGENTranslateDocDescription : CommandDescription<
  "kgen-translate", "-emit-mojo-doc",
  "Emits the documentation of a Mojo file or package as JSON.",
  "Generates the same JSON as `mojo doc`, without the mojo tool.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>;

def emit_mojo_doc : EmitMojoDocFlag;

def validate_doc_strings : Flag<["--", "-"], "validate-doc-strings">,
  Group<DiagnosticOptionGroup>, Flags<[HelpHidden]>,
  HelpText<"Deprecated: use -Werror instead.">;
