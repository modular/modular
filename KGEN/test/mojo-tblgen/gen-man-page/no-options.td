//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: %mojo-tblgen -gen-man-page %s | FileCheck %s

// RUN: %mojo-tblgen -gen-man-page %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "KGEN/DriverCommand.td"

// CHECK: .SH "SYNOPSIS"
// CHECK-NEXT: \fBmojo\-tblgen\-test no\-options\fR \fIpath...\fR
def : CommandDescription<
  "mojo-tblgen-test", "no-options", "Test of description with no options",
  "Tests that a man page can be emitted for a command with no options."
  "The synopsis should not include '[options]' in between the command name "
  "and the metavar name.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/true>;
