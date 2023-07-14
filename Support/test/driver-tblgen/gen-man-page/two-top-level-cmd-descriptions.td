//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: not %driver-tblgen -gen-man-page %s 2>&1 | FileCheck %s
// CHECK: error: cannot define a second top-level 'CommandDescription' record

include "Support/Driver/DriverCommand.td"

def TopLevelDescriptionOne : CommandDescription<
  "exec", /*subcommand=*/"", "Test that two top-level commands result in error",
  "This is a top-level driver record, since it doesn't have a subcommand name.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>;

def TopLevelDescriptionTwo : CommandDescription<
  "exec", /*subcommand=*/"", "Test that two top-level commands result in error",
  "Two top-level driver records cannot exist in the same set of records.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>;
