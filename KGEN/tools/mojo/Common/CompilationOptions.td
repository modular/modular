//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a common set of compilation options for setting up the
// compilation environment for Mojo programs. The index of the option groups is
// set to multiples of 5, to allow other option groups to be inserted before or
// after each.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<5>;

def no_optimization : Flag<["--", "-"], "no-optimization">,
  Group<CompilationOptionGroup>, Index<0>,
  HelpText<"Disables compiler optimizations. This might reduce the "
           "amount of time it takes to compile the Mojo source file. It might "
           "also reduce the runtime performance of the compiled executable.">;
def : Flag<["-"], "O0">, Group<CompilationOptionGroup>, Index<0>,
  Alias<no_optimization>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<1>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

def D : Separate<["-"], "D">, Group<CompilationOptionGroup>, Index<2>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Defines a named value that can be used from within the Mojo source "
           "file being executed. For example, `-D foo=42` defines a name `foo` "
           "that, when queried with the `sys.param_env` module from within the "
           "Mojo program, would yield the compile-time value `42`.">;

//===----------------------------------------------------------------------===//
// Experimental Compilation Options
//===----------------------------------------------------------------------===//

def ExperimentalCompilationOptionGroup :
  OptionGroup<"Experimental compilation options">, Index<15>;

defm debug_level : CommandOption<["--", "-"], "debug-level",
  ExperimentalCompilationOptionGroup, Index<0>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of debug info to use at compilation. The value must be one "
    "of: `none` (the default value), `line-tables`, or `full`. Please note "
    "that there are issues when generating debug info for some Mojo programs "
    "that have yet to be addressed."
  >>;
def : Flag<["-"], "g">, Group<ExperimentalCompilationOptionGroup>, Index<1>,
  Alias<debug_level>, AliasArgs<["full"]>;

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
