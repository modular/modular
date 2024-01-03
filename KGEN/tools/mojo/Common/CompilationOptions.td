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
  HelpText<"Disables compiler optimizations. This might reduce the "
           "amount of time it takes to compile the Mojo source file. It might "
           "also reduce the runtime performance of the compiled executable.">;
def : Flag<["-"], "O0">, Group<CompilationOptionGroup>, Index<0>,
  Alias<no_optimization>;

defm target_triple : CommandOption<["--", "-"], "target-triple",
  CompilationOptionGroup, Index<1>, MetaVarName<"TRIPLE">,
  HelpText<"Sets the compilation target triple. Defaults to the host target.">>;

defm target_cpu : CommandOption<["--", "-"], "target-cpu",
  CompilationOptionGroup, Index<2>, MetaVarName<"CPU">,
  HelpText<"Sets the compilation target CPU. Defaults to the host CPU.">>;

defm target_features : CommandOption<["--", "-"], "target-features",
  CompilationOptionGroup, Index<3>, MetaVarName<"FEATURES">,
  HelpText<"Sets the compilation target CPU features. Defaults to the host "
           "features.">>;

defm march : CommandOption<["--", "-"], "march", CompilationOptionGroup,
  Index<4>, MetaVarName<"ARCHITECTURE">,
  HelpText<"Sets the architecture for which to generate code.">>;

defm mcpu : CommandOption<["--", "-"], "mcpu", CompilationOptionGroup, Index<5>,
  MetaVarName<"CPU">, HelpText<"Sets the CPU for which to generate code.">>;

defm mtune : CommandOption<["--", "-"], "mtune", CompilationOptionGroup,
  Index<6>, MetaVarName<"TUNE">,
  HelpText<"Sets the CPU for which to tune code.">>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<7>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

def D : Separate<["-"], "D">, Group<CompilationOptionGroup>, Index<8>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Defines a named value that can be used from within the Mojo source "
           "file being executed. For example, `-D foo=42` defines a name `foo` "
           "that, when queried with the `sys.param_env` module from within the "
           "Mojo program, would yield the compile-time value `42`.">;

def parsing_stdlib : Flag<["--", "-"], "parsing-stdlib">,
  Group<CompilationOptionGroup>, Index<9>, Flags<[HelpHidden]>,
  HelpText<"Parses the input file(s) as the Mojo standard library.">;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<2>;

def warn_missing_dog_strings : Flag<["--", "-"], "warn-missing-doc-strings">,
  Group<DiagnosticOptionGroup>, Index<0>,
  HelpText<"Emits warnings for missing or partial docstrings.">;

defm max_notes : CommandOption<["--", "-"], "max-notes-per-diagnostic",
  DiagnosticOptionGroup, Index<1>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">>;

//===----------------------------------------------------------------------===//
// Experimental Compilation Options
//===----------------------------------------------------------------------===//

def ExperimentalCompilationOptionGroup :
  OptionGroup<"Experimental compilation options">, Index<3>;

defm debug_level : CommandOption<["--", "-"], "debug-level",
  ExperimentalCompilationOptionGroup, Index<0>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of debug info to use at compilation. The value must be one "
    "of: `none` (the default value), `line-tables`, or `full`. Please note "
    "that there are issues when generating debug info for some Mojo programs "
    "that have yet to be addressed."
  >>;

defm sanitize : CommandOption<["--", "-"], "sanitize",
  ExperimentalCompilationOptionGroup, Index<1>, MetaVarName<"CHECK">,
  HelpText<
    "Turns on runtime checks. The following values are supported: `address` "
    "(detects memory issues), and `thread` (detects multi-threading issues). "
    "Please note that these checks are not currently supported when executing "
    "Mojo programs."
  >>;

defm debug_info_language : CommandOption<["--", "-"], "debug-info-language",
  ExperimentalCompilationOptionGroup, Index<2>, MetaVarName<"LANGUAGE">,
  HelpText<
    "Sets the language to emit as part of the debug info. The supported "
    "languages are: `Mojo`, and `C`. `C` is the default, and is useful to "
    "enable rudimentary debugging and binary introspection in tools "
    "that don't understand Mojo."
  >>;

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
