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

defm optimization_level : CommandOption<["--", "-"], "optimization-level",
  CompilationOptionGroup, Index<0>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of optimization to use at compilation. The value must be "
    "a number between 0 and 3. The default is 3."
  >>;
def : JoinedOrSeparate<["-"], "O">, Group<CompilationOptionGroup>,
  Index<1>, Alias<optimization_level>;
def : Flag<["--", "-"], "no-optimization">, Group<CompilationOptionGroup>,
  Index<2>, Alias<optimization_level>, AliasArgs<["0"]>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<3>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

def D : Separate<["-"], "D">, Group<CompilationOptionGroup>, Index<4>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Defines a named value that can be used from within the Mojo source "
           "file being executed. For example, `-D foo=42` defines a name `foo` "
           "that, when queried with the `sys.param_env` module from within the "
           "Mojo program, would yield the compile-time value `42`.">;

defm debug_level : CommandOption<["--", "-"], "debug-level",
  CompilationOptionGroup, Index<5>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of debug info to use at compilation. The value must be one "
    "of: `none` (the default value), `line-tables`, or `full`. Please note "
    "that there are issues when generating debug info for some Mojo programs "
    "that have yet to be addressed."
  >>;

def : Flag<["-"], "g">, Group<CompilationOptionGroup>, Index<6>,
  Alias<debug_level>, AliasArgs<["full"]>;

defm num_threads : CommandOption<["--", "-"], "num-threads",
  CompilationOptionGroup, Index<7>, MetaVarName<"NUM">,
  HelpText<
    "Sets the maximum number of threads to use for compilation. "
    "The default is 0 (use all available threads)."
  >>;

def : JoinedOrSeparate<["-"], "j">, Group<CompilationOptionGroup>,
  Index<8>, Alias<num_threads>;


//===----------------------------------------------------------------------===//
// Experimental Compilation Options
//===----------------------------------------------------------------------===//

def ExperimentalCompilationOptionGroup :
  OptionGroup<"Experimental compilation options">, Index<15>;

defm sanitize : CommandOption<["--", "-"], "sanitize",
  ExperimentalCompilationOptionGroup, Index<1>, MetaVarName<"CHECK">,
  HelpText<
    "Turns on runtime checks. The following values are supported: `address` "
    "(detects memory issues), and `thread` (detects multi-threading issues)."
  >>;

def shared_libasan : Flag<["--", "-"], "shared-libasan">,
  Group<ExperimentalCompilationOptionGroup>, Index<2>,
  HelpText<
    "Dynamically link the address sanitizer runtime. Requires address "
    "sanitization turned on with `--sanitize` option."
  >;

defm external_libasan : CommandOptionHidden<["--", "-"], "external-libasan",
  ExperimentalCompilationOptionGroup, Index<3>, MetaVarName<"FILE">,
  HelpText<
    "Uses the provided file as libasan instead of the default provided by "
    "the system. This file must provide a compatible interface as libasan, "
    "and must be present at both compile time and run time of the program."
  >>;

defm debug_info_language : CommandOption<["--", "-"], "debug-info-language",
  ExperimentalCompilationOptionGroup, Index<3>, MetaVarName<"LANGUAGE">,
  HelpText<
    "Sets the language to emit as part of the debug info. The supported "
    "languages are: `Mojo`, and `C`. `C` is the default, and is useful to "
    "enable rudimentary debugging and binary introspection in tools "
    "that don't understand Mojo."
  >>;

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
