//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %driver-tblgen -gen-markdown %s | FileCheck %s

// RUN: %driver-tblgen -gen-markdown %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that options are output in Markdown",
  "This command contains an option group with options.",
  /*inputMetaVarName=*/"input", /*requiresInput=*/false,
  /*variadicInput=*/false>;

// CHECK: ## Commands
// CHECK: `subcommand` — Test that subcommands are output in Markdown
def : CommandDescription<
  "driver-tblgen-test", "subcommand",
  "Test that subcommands are output in Markdown",
  "This is the description of a subcommand.",
  /*inputMetaVarName=*/"input", /*requiresInput=*/false,
  /*variadicInput=*/false>, Index<0>;

// CHECK: ## Options
// CHECK: ### First Options
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

// CHECK: #### `--foo <FOO>`, `-F<ALSO-FOO>`
// CHECK: Takes a foo and does something with it.
def foo : JoinedOrSeparate<["--", "-"], "foo">, Group<FirstOptionGroup>,
  Index<0>, MetaVarName<"FOO">,
  HelpText<"Takes a foo and does something with it.">;
def : Joined<["-"], "F">, Group<FirstOptionGroup>, Index<0>,
  MetaVarName<"ALSO-FOO">, Alias<foo>;
