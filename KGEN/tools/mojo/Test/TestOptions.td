//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Test/TestDescription.td"
include "Support/Driver/CommonOptions.td"
include "Common/DiagnosticFormatOptions.td"
include "../Common/CompilationOptions.td"
include "../Common/DiagnosticOptions.td"
include "../Common/TargetOptions.td"
include "../Common/StandaloneDebugOptions.td"

//===----------------------------------------------------------------------===//
// Collection Options
//===----------------------------------------------------------------------===//

def CollectionOptionGroup : OptionGroup<"Collection options">, Index<0>;

def collect_only : Flag<["--"], "collect-only">,
  Group<CollectionOptionGroup>, Index<0>,
  HelpText<"Only collect tests, don't execute them.">;
def : Flag<["--"], "co">, Group<CollectionOptionGroup>, Index<0>,
  Alias<collect_only>;

def keep_entrypoint : Flag<["--"], "keep-entrypoint">,
  Group<CollectionOptionGroup>, Index<1>,
  HelpText<"Keep the test entrypoint for later debugging.">,
  Flags<[HelpHidden]>;

defm entrypoint_path : CommandOptionHidden<["--"], "entrypoint-path",
  CollectionOptionGroup, Index<2>, MetaVarName<"PATH">,
  HelpText<"Override the location for the entrypoint executable.">>;

def no_execute : Flag<["--"], "no-execute">,
  Group<CollectionOptionGroup>, Index<3>,
  HelpText<"Build the test entrypoint but do not execute tests.">,
  Flags<[HelpHidden]>;

//===----------------------------------------------------------------------===//
// Test Run Options
//===----------------------------------------------------------------------===//

def TestRunOptionGroup : OptionGroup<"Test run options">, Index<1>;

def debug : Flag<["--"], "debug">,
  Group<TestRunOptionGroup>, Index<0>,
  HelpText<"Launch a debugger session with `mojo debug`. This is not supported for docstring tests. Most debug flags are supported, including `--rpc`.">;
