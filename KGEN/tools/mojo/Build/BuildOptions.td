//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "BuildDescription.td"

include "Common/CompilationOptions.td"
include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/TargetOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">, HelpText<
    "Sets the path and filename for the executable output. By default, it "
    "outputs the executable to the same location as the Mojo file, with the "
    "same name and no extension."
  >>;

defm emitted_file_type: CommandOptionWithValues<
  ["--"], "emit", OutputOptionGroup, Index<1>,
  MetaVarName<"FILE_TYPE">,
  HelpText<"The type of output file to generate."
            "\n\n"
            "* `exe` (default): emit an executable binary file.\n"
            "* `shared-lib`: emit a shared (dynamic) library.\n"
            "* `object`: (EXPERIMENTAL) emit a single object file.\n"
            "* `llvm`: emit LLVM IR.\n"
            "* `asm`: emit target assembly.">,
  Values<"exe,shared-lib,object,llvm,asm">
>;

def generate_python_extension_module :  Flag<["--"], "gen-py">,
  Group<OutputOptionGroup>, Index<2>,
  HelpText<"Whether to generate a Python extension module (dynamic library). Defaults to false.">,
  Flags<[HelpHidden]>;
