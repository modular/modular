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
// This file defines the fix-it options, for subcommands that parse through
// invokeMojoParser and so can apply or export the fix-its it collects.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_FIXITOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_FIXITOPTIONS_TD

include "Common/DiagnosticOptions.td"

def fixit : Flag<["--", "-"],
  "experimental-fixit">, Group<DiagnosticOptionGroup>, Index<8>,
  HelpText<"Automatically apply fix-its to the code, and rerun the command "
           "again after the fix-its are applied. WARNING: this feature is "
           "highly experimental and may result in irreversible data loss.">,
  MarshallingInfoFlag<DriverOpts<"applyFixIts">>;

defm export_fixit : CommandOption<["--", "-"], "experimental-export-fixit",
  DiagnosticOptionGroup, Index<9>, MetaVarName<"YAML_FILE">,
  HelpText<"Export fix-its to a YAML file in clang-tidy format instead of "
           "applying them directly. The file can be applied using "
           "'clang-apply-replacements'. WARNING: this feature is highly "
           "experimental.">>,
  MarshallingInfoString<DriverOpts<"exportFixItsPath">>;

#endif // KGEN_TOOLS_MOJO_COMMON_FIXITOPTIONS_TD
