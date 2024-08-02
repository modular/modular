//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Test/TestDescription.td"
include "Common/DiagnosticFormatOptions.td"
include "Support/Driver/CommonOptions.td"

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

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<2>;

// TODO: Remove this when we can infer additional import directories from the
// test file's location.
def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;
