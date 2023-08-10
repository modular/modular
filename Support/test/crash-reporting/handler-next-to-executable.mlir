// RUN: rm -rf %t
// RUN: mkdir -p %t/test-bin
// RUN: cp %crash-report-path-info %t/test-bin/crash-report-path-info
// RUN: cp %modular-crashpad-handler %t/test-bin/modular-crashpad-handler
// RUN: env -i %t/test-bin/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}test-bin{{[\\/]}}modular-crashpad-handler
