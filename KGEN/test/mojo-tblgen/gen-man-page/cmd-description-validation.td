//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %mojo-tblgen -gen-man-page %s 2>&1 >/dev/null | FileCheck %s

include "Support/Driver/DriverCommand.td"

// CHECK: warning: ignoring description
def : CommandDescription<
  "mojo-tblgen-test", "first-command-descriptions",
  "Test that this record is ignored",
  "Multiple command description records should not appear in a single input, "
  "since each one is meant to describe the command for which documentation is "
  "being generated. Since another description record appears below this one, "
  "this record is ignored.", /*inputMetaVarName=*/"path",
  /*requiresInput=*/true, /*variadicInput=*/false>;

// CHECK: warning: command executable name should not be empty
// CHECK: warning: command summary should begin with a capital letter
// CHECK: warning: command summary should not end with a period
// CHECK: warning: command description should begin with a capital letter
// CHECK: warning: command description should end with a period
// CHECK: warning: command input metavar name should be lowercase
def : CommandDescription<
  "", "second-command-description", "test that this record is not ignored.",
  "this record appears after the first, which will be ignored. Technically, "
  "this record will be selected, but, ideally users will not ignore warnings "
  "and ensure they only define a single description record.\n\n"
  "This record also fails several of mojo-tblgen's validation checks",
  /*inputMetaVarName=*/"Path", /*requiresInput=*/true, /*variadicInput=*/false>;
