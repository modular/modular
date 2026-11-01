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
// The options of the translations that run through mlirTranslateMain.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATEMLIROPTIONS_TD
#define KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATEMLIROPTIONS_TD

include "Common/DeveloperOptions.td"
include "KGENTranslateCommonOptions.td"

def mlir_print_debuginfo : Flag<["--", "-"], "mlir-print-debuginfo">,
  Group<OutputOptionGroup>,
  HelpText<"Prints debug locations in the MLIR output.">;

def mlir_print_op_generic : Flag<["--", "-"], "mlir-print-op-generic">,
  Group<OutputOptionGroup>,
  HelpText<"Prints the MLIR output in the generic operation form.">;

def kgen_print_inline_type_values : Flag<["--", "-"],
  "kgen-print-inline-type-values">,
  Group<OutputOptionGroup>,
  HelpText<"Prints KGEN type values inline instead of as aliases.">;

defm output_split_marker : CommandOption<["--", "-"], "output-split-marker",
  OutputOptionGroup, MetaVarName<"MARKER">,
  HelpText<"Joins the outputs of the chunks of a split input with MARKER.">>,
  MarshallingInfoString<TranslateOpts<"outputSplitMarker">>;

//===----------------------------------------------------------------------===//
// Testing Options
//===----------------------------------------------------------------------===//

def TestingOptionGroup : OptionGroup<"Testing options">, Index<20>;

def split_input_file : Flag<["--", "-"], "split-input-file">,
  Group<TestingOptionGroup>,
  HelpText<"Splits the input at `// -----` lines and translates each chunk "
           "on its own.">,
  MarshallingInfoFlag<TranslateOpts<"splitInputFile">>;
def split_input_file_eq : Joined<["--", "-"], "split-input-file=">,
  Group<TestingOptionGroup>, MetaVarName<"MARKER">,
  HelpText<"Splits the input at lines of MARKER instead.">,
  MarshallingInfoString<TranslateOpts<"inputSplitMarker">>;

def verify_diagnostics_eq : Joined<["--", "-"], "verify-diagnostics=">,
  Group<TestingOptionGroup>, MetaVarName<"LEVEL">,
  HelpText<"Checks the emitted diagnostics against the `expected-*` comments "
           "in the input: `all` of them, or `only-expected` ones.">,
  MarshallingInfoChoice<TranslateOpts<"verifyDiagnosticsLevel">,
                        ["all", "only-expected"], ["All", "OnlyExpected"]>;
def verify_diagnostics : Flag<["--", "-"], "verify-diagnostics">,
  Group<TestingOptionGroup>, Alias<verify_diagnostics_eq>, AliasArgs<["all"]>,
  HelpText<"Checks the emitted diagnostics against the `expected-*` comments "
           "in the input.">;

def allow_unregistered_dialect : Flag<["--", "-"],
  "allow-unregistered-dialect">,
  Group<TestingOptionGroup>,
  HelpText<"Accepts operations of dialects that are not registered.">,
  MarshallingInfoFlag<TranslateOpts<"allowUnregisteredDialects">>;

def error_diagnostics_only : Flag<["--", "-"], "error-diagnostics-only">,
  Group<TestingOptionGroup>,
  HelpText<"Reports only error diagnostics.">,
  MarshallingInfoFlag<TranslateOpts<"errorDiagnosticsOnly">>;

#endif // KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATEMLIROPTIONS_TD
