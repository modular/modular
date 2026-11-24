// RUN: mef-translate -export-mef %s -o %t
// RUN: od -t x1 %t | FileCheck %s

// Empty file.  Check to make sure the magic number is right.

// CHECK: 0000000 ee f0
