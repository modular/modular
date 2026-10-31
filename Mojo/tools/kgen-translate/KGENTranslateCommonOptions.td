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
// The options that every kgen-translate translation shares.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATECOMMONOPTIONS_TD
#define KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATECOMMONOPTIONS_TD

include "Common/Marshalling.td"
include "Support/Driver/CommonOptions.td"
include "Support/Driver/DriverCommand.td"

// A field of kgen-translate's `TranslateOptions options` being built.
class TranslateOpts<string member>
    : KeyPathAndMacro<"options.", member, "MOJO_">;

//===----------------------------------------------------------------------===//
// Translation Options
//===----------------------------------------------------------------------===//

def TranslationOptionGroup : OptionGroup<"Translations">, Index<0>,
  HelpText<"Exactly one translation must be given.">;

// Each translation's table defines only the flags it accepts.
class ImportMojoFlag : Flag<["--", "-"], "import-mojo">,
  Group<TranslationOptionGroup>,
  HelpText<"Imports Mojo source as MLIR.">;

class EmitMojoDocFlag : Flag<["--", "-"], "emit-mojo-doc">,
  Group<TranslationOptionGroup>,
  HelpText<"Emits the documentation of a Mojo file or package as JSON.">;

class MLIRToLLVMIRFlag : Flag<["--", "-"], "mlir-to-llvmir">,
  Group<TranslationOptionGroup>,
  HelpText<"Translates MLIR to LLVM IR.">;

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<1>;

defm o : CommandOption<["-"], "o", OutputOptionGroup,
  MetaVarName<"PATH">,
  HelpText<"Sets the output path. Defaults to stdout.">>,
  MarshallingInfoString<TranslateOpts<"outputPath">>;

#endif // KGEN_TOOLS_KGEN_TRANSLATE_KGENTRANSLATECOMMONOPTIONS_TD
