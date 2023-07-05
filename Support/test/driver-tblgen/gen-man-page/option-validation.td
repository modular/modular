//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %driver-tblgen -gen-man-page %s 2>&1 >/dev/null | FileCheck %s

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  "driver-tblgen-test", "option-validation", "Test option validation",
  "Test that warnings are emitted when options do not include help text, or "
  "when they do not match the style we expect.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>;

def FunOptionGroup : OptionGroup<"Fun Options">;

// CHECK: warning: help text for publicly visible option 'chute' should not be empty
// CHECK: warning: option 'chute' takes a value, but does not define a metavar name for that value
def chute : Separate<["-"], "chute">, Group<FunOptionGroup>, Index<0>, Flags<[]>;

// CHECK: warning: option 'ladder' metavar name should not be empty
def ladder : Joined<["--"], "ladder">, Group<FunOptionGroup>, Index<1>,
  Flags<[]>, HelpText<"The ladder to use.">, MetaVarName<"">;
// CHECK: warning: alias 'l' has no index
def l : Joined<["-"], "l">, Group<FunOptionGroup>, Flags<[]>, Alias<ladder>;

// CHECK: warning: option 'slide' metavar name should be uppercase
// CHECK: warning: option 'slide' has no index
def slide : JoinedOrSeparate<["--", "-"], "slide">, Group<FunOptionGroup>,
  Flags<[]>, HelpText<"The slide to use.">, MetaVarName<"path">;

// CHECK: warning: group 'Fun Options' has no index
