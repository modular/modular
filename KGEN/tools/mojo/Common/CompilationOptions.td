//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a common set of compilation options for setting up the
// compilation environment for Mojo programs. The index of the option groups is
// set to 1 and 2, to allow options for output to be defined first.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<1>;

def no_optimization : Flag<["--", "-"], "no-optimization">,
  Group<CompilationOptionGroup>, Index<0>,
  HelpText<"Whether to disable compiler optimizations. This may reduce the "
           "amount of time it takes to compile the Mojo source file. It may "
           "also reduce the runtime performance of the compiled executable.">;
def : Flag<["-"], "O0">, Group<CompilationOptionGroup>, Index<0>,
  Alias<no_optimization>;

def debug_level : Separate<["--", "-"], "debug-level">,
  Group<CompilationOptionGroup>, Index<1>, MetaVarName<"LEVEL">,
  HelpText<"The level of debug info to use during compilation. The value must "
           "be one of: `none` (the default value), `line-tables`, or `full`.">;

def target_triple : Separate<["--", "-"], "target-triple">,
  Group<CompilationOptionGroup>, Index<2>, MetaVarName<"TRIPLE">,
  HelpText<"Compilation target triple. Defaults to the host target.">;

def target_cpu : Separate<["--", "-"], "target-cpu">,
  Group<CompilationOptionGroup>, Index<3>, MetaVarName<"CPU">,
  HelpText<"Compilation target CPU. Defaults to the host CPU.">;

def target_features : Separate<["--", "-"], "target-features">,
  Group<CompilationOptionGroup>, Index<4>, MetaVarName<"FEATURES">,
  HelpText<"Compilation target CPU features. Defaults to the host features.">;

def march : Separate<["-", "--"], "march">, Group<CompilationOptionGroup>,
  Index<5>, MetaVarName<"ARCHITECTURE">,
  HelpText<"Architecture to generate code for.">;

def mcpu : Separate<["-", "--"], "mcpu">, Group<CompilationOptionGroup>,
  Index<6>, MetaVarName<"CPU">,
  HelpText<"CPU to generate code for.">;

def mtune : Separate<["-", "--"], "mtune">, Group<CompilationOptionGroup>,
  Index<7>, MetaVarName<"TUNE">,
  HelpText<"CPU to tune code for.">;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<7>,
  MetaVarName<"PATH">,
  HelpText<"Append the given path to the list of directories to search for "
           "imported Mojo files.">;

def D : Separate<["-"], "D">, Group<CompilationOptionGroup>, Index<9>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Define a named value that can be used from within the Mojo source "
           "file being executed. For example, `-D foo=42` defines a name `foo` "
           "that, when queried with the `ParamEnv` module from within the Mojo "
           "program, would yield the compile-time value `42`.">;

def sanitize : Separate<["--", "-"], "sanitize">, Group<CompilationOptionGroup>,
  Index<10>, MetaVarName<"CHECK">,
  HelpText<
    "Turn on runtime checks. The following values are supported: `address` "
    "(detects memory issues), and `thread` (detects multi-threading issues)."
  >;

def parsing_stdlib : Flag<["--", "-"], "parsing-stdlib">,
  Group<CompilationOptionGroup>, Index<11>,
  HelpText<"Parse the input file(s) as the Mojo standard library.">;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<2>;

def warn_missing_dog_strings : Flag<["--", "-"], "warn-missing-doc-strings">,
  Group<DiagnosticOptionGroup>, Index<0>,
  HelpText<"Emit warnings for missing or partial doc strings.">;

def max_notes : Separate<["--", "-"], "max-notes-per-diagnostic">,
  Group<DiagnosticOptionGroup>, Index<1>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">;

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
