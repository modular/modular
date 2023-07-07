//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "../OptionsBase.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// run Command
//===----------------------------------------------------------------------===//

def RunDescription : MojoCommandDescription<
  "run", "Build and execute a Mojo file",
  "Compiles the Mojo file at the given path and immediately executes it.\n"
  "\n"
  "Options for this command itself, such as the ones listed below, must appear "
  "before the input file `path` argument. Any command line arguments that "
  "appear after the Mojo source file `path` are interpreted as arguments for "
  "that Mojo program.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>;

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation Options">, Index<0>;

def targetTriple : Separate<["--", "-"], "target-triple">,
  Group<CompilationOptionGroup>, Index<0>, MetaVarName<"TRIPLE">,
  HelpText<"Compilation target triple. Defaults to the host target.">;

def targetCpu : Separate<["--", "-"], "target-cpu">,
  Group<CompilationOptionGroup>, Index<1>, MetaVarName<"CPU">,
  HelpText<"Compilation target CPU. Defaults to the host CPU.">;

def targetFeatures : Separate<["--", "-"], "target-features">,
  Group<CompilationOptionGroup>, Index<2>, MetaVarName<"FEATURES">,
  HelpText<"Compilation target CPU features. Defaults to the host features.">;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<3>,
  MetaVarName<"PATH">,
  HelpText<"Append the given path to the list of directories to search for "
           "included Mojo files.">;

def L : JoinedOrSeparate<["-"], "L">, Group<CompilationOptionGroup>, Index<4>,
  MetaVarName<"PATH">,
  HelpText<"Append the given path to the list of directories to search for "
           "linked libraries.">;

def no_optimization : Flag<["--", "-"], "no-optimization">,
  Group<CompilationOptionGroup>, Index<5>,
  HelpText<"Whether to disable compiler optimizations. This may reduce the "
           "amount of time it takes to compile the Mojo source file. It may "
           "also reduce the runtime performance of the compiled executable.">;
def : Flag<["-"], "O0">, Group<CompilationOptionGroup>, Index<0>,
  Alias<no_optimization>;

//===----------------------------------------------------------------------===//
// Validation Options
//===----------------------------------------------------------------------===//

def ValidationOptionGroup : OptionGroup<"Validation Options">, Index<1>;

def doc_validate : Flag<["--", "-"], "doc-validate">,
  Group<ValidationOptionGroup>, Index<0>,
  HelpText<"Validate doc strings as they are parsed during compilation. When "
           "enabled, warning diagnostics are emitted as invalid doc strings "
           "are parsed.">;
