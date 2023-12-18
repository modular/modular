//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %driver-tblgen -gen-help-text %s | FileCheck %s

// RUN: %driver-tblgen -gen-help-text %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

// CHECK: driver-tblgen-test [options] <inputs...>
// CHECK-NEXT: driver-tblgen-test <input>
// CHECK-NEXT: driver-tblgen-test
def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that options are output in help text.",
  "This command contains an option group with options.",
  [
    UsageInfo<"options", "inputs", /*variadicInput=*/true>,
    UsageInfo<"", "input", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;

// CHECK: COMMANDS
// CHECK: subcommand — Test that subcommands are output in help text
def : CommandDescription<
  "driver-tblgen-test", "subcommand",
  "Test that subcommands are output in help text.",
  "This is the description of a subcommand.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<0>;

// CHECK: OPTIONS
// CHECK: First Options
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

// CHECK: --foo <FOO>, -F<ALSO-FOO>
// CHECK-NEXT: Takes a foo and does something with it.
defm foo : CommandOption<["--", "-"], "foo", FirstOptionGroup, Index<0>,
  MetaVarName<"FOO">, HelpText<"Takes a foo and does something with it.">>;
def : Joined<["-"], "F">, Group<FirstOptionGroup>, Index<1>,
  MetaVarName<"ALSO-FOO">, Alias<foo>;

// Values do not make their way into --help output.
// CHECK: --bar <BAR>
// CHECK-NEXT: Takes a bar and does something with it.
defm bar : CommandOptionWithValues<["--", "-"], "bar", FirstOptionGroup,
  Index<2>, MetaVarName<"BAR">,
  HelpText<"Takes a bar and does something with it.">, Values<"a,b,c">>;
