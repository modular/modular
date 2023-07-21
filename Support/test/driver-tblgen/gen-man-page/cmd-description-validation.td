//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %driver-tblgen -gen-man-page %s 2>&1 >/dev/null | FileCheck %s

include "Support/Driver/DriverCommand.td"

// CHECK: warning: command executable name should not be empty
// CHECK: warning: command summary should begin with a capital letter
// CHECK: warning: command summary should end with a period
// CHECK: warning: command description should begin with a capital letter
// CHECK: warning: command description should end with a period
// CHECK: warning: usage options name should be lowercase
// CHECK: warning: usage input metavar name should be lowercase
def TopLevelDescription : CommandDescription<
  /*executable=*/"", /*subcommand=*/"", "test this record produces warnings",
  "this is a top-level driver record, since it does not have a subcommand name",
  [
    UsageInfo<"Options", "path", /*variadicInput=*/false>,
    UsageInfo<"options", "patH", /*variadicInput=*/false>,
  ]>;

// CHECK: warning: command executable name should not be empty
// CHECK: warning: command summary should begin with a capital letter
// CHECK: warning: command summary should end with a period
// CHECK: warning: command description should begin with a capital letter
// CHECK: warning: command description should end with a period
// CHECK: warning: command should include at least one usage
// CHECK: warning: subcommand 'sub-one' has no index
def SubcommandOneDescription : CommandDescription<
  "", "sub-one", "test this record also produces warnings",
  "this is a subcommand record", /*usages=*/[]>;

// CHECK: warning: subcommand '{{.*}}' has index 1, which has already been used
// CHECK: note: subcommand '{{.*}}' has already been defined with index 1 here
def SubcommandTwoDescription : CommandDescription<
  "exec", "sub-two", "Test this record is warned about because of its index.",
  "This record shares an index with the one below.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>, Index<1>;
def SubcommandThreeDescription : CommandDescription<
  "exec", "sub-three", "Test this record is warned about because of its index.",
  "This record shares an index with the one above.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>, Index<1>;
