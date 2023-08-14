// UNSUPPORTED: windows
// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t/fake-path %t/test-bin
// RUN: cp %crash-report-path-info %t/test-bin/crash-report-path-info
// RUN: cp %modular-crashpad-handler %t/test-bin/modular-crashpad-handler
// RUN: cp %modular-crashpad-handler %t/fake-path/modular-crashpad-handler
// RUN: env -i PATH=%t/fake-path %t/test-bin/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}test-bin{{[\\/]}}modular-crashpad-handler
