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

def : CommandDescription<
  "mojo-tblgen-test", "options", "Test that options are output in a man page",
  "This command contains several option groups, each of which has several "
  "options, along with aliases.", /*inputMetaVarName=*/"input",
  /*requiresInput=*/false, /*variadicInput=*/false>;

// CHECK: .SH "OPTIONS"
// CHECK: .SS "First Options"
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

// Note that the following options have difference indices than their source
// order. We test here that they appear in ascending index order.
// CHECK: \fB\-bar\fR, \fB\-\-baz\fR, \fB\-\-bazzer\fR, \fB\-\-bazzest\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Turns on bar support.
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
// CHECK: \fB\-\-foo\fR \fIFOO\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Takes a foo and does something with it.
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
def : JoinedOrSeparate<["--", "-"], "foo">, Group<FirstOptionGroup>, Index<1>,
  Flags<[]>, MetaVarName<"FOO">, HelpText<"Takes a foo and does something with it.">;
def bar : Flag<["-"], "bar">, Group<FirstOptionGroup>, Index<0>, Flags<[]>,
  HelpText<"Turns on bar support.">;
def : Flag<["--"], "bazzest">, Group<FirstOptionGroup>, Index<2>, Flags<[]>,
  Alias<bar>;
def : Flag<["--"], "baz">, Group<FirstOptionGroup>, Index<0>, Flags<[]>,
  Alias<bar>;
def : Flag<["--"], "bazzer">, Group<FirstOptionGroup>, Index<1>, Flags<[]>,
  Alias<bar>;

// CHECK: .SS "Second Options"
// CHECK-NEXT: Another \-\- potentially less useful \-\- set of options.
// CHECK-NEXT: .sp
def SecondOptionGroup : OptionGroup<"Second Options">, Index<1>,
  HelpText<"Another -- potentially less useful -- set of options.">;

// CHECK: \fB\-I\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Specify the "I", as in "igloo."
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
def : Joined<["-"], "I">, Group<SecondOptionGroup>, Index<0>, Flags<[]>,
  HelpText<"Specify the \"I\", as in \"igloo.\"">, MetaVarName<"IGLOO">;
