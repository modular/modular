//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: not %driver-tblgen -gen-man-page %s 2>&1 >/dev/null | FileCheck %s

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  "driver-tblgen-test", "option-validation", "Test option validation.",
  "Test that errors are emitted when options do not include help text, or "
  "when they do not match the style we expect.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>;

def FunOptionGroup : OptionGroup<"Fun Options">;

// CHECK: error: help text for publicly visible option 'chute' should not be empty
// CHECK: error: option 'chute' takes a value, but does not define a metavar name for that value
def chute : Separate<["-"], "chute">, Group<FunOptionGroup>, Index<0>;

// CHECK: error: option 'ladder' metavar name should not be empty
def ladder : Joined<["--"], "ladder">, Group<FunOptionGroup>, Index<1>,
  HelpText<"The ladder to use.">, MetaVarName<"">;
def stool : Flag<["--"], "stool">, Group<FunOptionGroup>, Index<2>,
  HelpText<"Whether to use a stool.">;
// CHECK: error: alias 's' has no index
def s : Joined<["-"], "s">, Group<FunOptionGroup>, Alias<stool>;

// CHECK: error: option 'slide' metavar name should be uppercase
// CHECK: error: option 'slide' has no index
def slide : JoinedOrSeparate<["--", "-"], "slide">, Group<FunOptionGroup>,
  HelpText<"The slide to use.">, MetaVarName<"path">;

// CHECK: error: group 'Fun Options' has no index
