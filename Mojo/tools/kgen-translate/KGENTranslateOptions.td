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

include "Common/CompilationOptions.td"
include "Common/DeveloperOptions.td"
include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/IncrParsingOptions.td"
include "Common/TargetOptions.td"
include "KGENTranslateMLIROptions.td"

def KGENTranslateDescription : DriverDescription<
  "kgen-translate", "Translates between Mojo, MLIR and LLVM IR.",
  "A testing tool that runs one translation on its input: importing Mojo "
  "source into MLIR, emitting Mojo documentation as JSON, or translating MLIR "
  "to LLVM IR. These are the options of -import-mojo; give a translation with "
  "--help to list the options of that translation.",
  [UsageInfo<"options", "file", /*variadicInput=*/false>]>;

// This table also selects the translation, so it accepts all three.
def import_mojo : ImportMojoFlag;
def emit_mojo_doc : EmitMojoDocFlag;
def mlir_to_llvmir : MLIRToLLVMIRFlag;

defm bytecode_output : CommandOption<["--", "-"], "bytecode-output",
  OutputOptionGroup, MetaVarName<"PATH">,
  HelpText<"Also writes the imported Mojo module as bytecode to PATH.">>,
  MarshallingInfoString<TranslateOpts<"bytecodeOutput">>;

def lsp: Flag<["--", "-"], "lsp">,
  Group<TestingOptionGroup>,
  HelpText<"Imports Mojo source the way the language server parses it.">,
  MarshallingInfoFlag<TranslateOpts<"lspMode">>;

def mojo_enable_prebuilt_packages : Flag<["--", "-"],
  "mojo-enable-prebuilt-packages">,
  Group<TestingOptionGroup>,
  HelpText<"Uses prebuilt packages when importing Mojo source.">,
  MarshallingInfoNegativeFlag<FrontendOpts<"disablePrebuiltPackages">>;

def use_mlir_diagnostics_eq : Joined<["--", "-"], "use-mlir-diagnostics=">,
  Group<TestingOptionGroup>, MetaVarName<"BOOL">,
  HelpText<"Sets whether diagnostics are reported through MLIR, which "
           "`--verify-diagnostics` needs. Defaults to `true`.">,
  MarshallingInfoCustom<FrontendOpts<"useMLIRDiagnostics">,
    "makeChoiceNormalizer<bool>({{\"true\", true}, {\"false\", false}})">;
