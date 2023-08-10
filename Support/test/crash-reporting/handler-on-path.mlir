// RUN: rm -rf %t
// RUN: mkdir -p %t/fake-path
// RUN: cp %crash-report-path-info %t/crash-report-path-info
// RUN: cp %modular-crashpad-handler %t/fake-path/modular-crashpad-handler
// RUN: env -i PATH=%t/fake-path %t/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}fake-path{{[\\/]}}modular-crashpad-handler
