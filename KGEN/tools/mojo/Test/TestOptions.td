//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Test/TestDescription.td"
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
