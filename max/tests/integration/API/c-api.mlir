// COM: Invalid path
// RUN: c-api-validator --paths non-existent-path 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// CHECK-ERROR: Invalid path: non-existent-path
