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
