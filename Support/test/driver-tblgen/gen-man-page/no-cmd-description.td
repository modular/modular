//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: not %driver-tblgen -gen-man-page %s 2>&1 | FileCheck %s
// CHECK: error: you must define a 'CommandDescription' record

include "Support/Driver/DriverCommand.td"
